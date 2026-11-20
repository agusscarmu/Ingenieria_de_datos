"""Las 17 características de variabilidad del ritmo cardíaco (VFC) de una ventana de 30 s, sobre la serie NN.

El código está copiado del de mi tesis doctoral. El método es el de Barrera y Vecino Schandy (Proyecto Final de Carrera,
ITBA, 2023; abajo, "el PFC"): las fórmulas del §4.5.1 (ec. 4.30 a 4.51). Las 17 son las de la tabla 1 de Liberczuk,
García, Barrera y Bonomini (SABI 2025; abajo, "el paper"). Sólo las calcula; no elige cuáles quedan.

**Una función pura por característica**, todas con el sufijo `_f` y la misma firma, `nombre_f(nn_ms, contiguo) ->
float`: `nn_ms` son los NN de la ventana en ms, en el orden del tiempo, y `contiguo[j]` dice si el NN `j` empieza donde
terminó el `j - 1` (el primero es siempre falso; en una serie cruda, los demás son todos verdaderos). Aceptan listas o
arreglos.
Sin disco ni pandas: la ventana entra y sale un número. Las 17, con el nombre de su columna entre paréntesis:

- **Dispersión:** `cv_f` (`P_cv`), `rmssd_f` (`P_RMSSD`), `nmasd_f` (`P_NMASD`), `ipd_f` (`P_IPD`, puntos de
  inflexión), `pnn50_f` (`PNN50`) y `pnn100_f` (`PNN100`).
- **Entropía y regularidad:** `samen_f` (`P_SamEn`), `ssamen_f` (`P_SSamEn`), `csamen_f` (`P_CSamEn`), `shen_f`
  (`P_ShEn`, Shannon del histograma) y `shan2_f` (`P_Shan2`, definición propia: el paper no la define).
- **Poincaré:** `nzpp1_f`, `nzpp2_f`, `nzpp3_f` (`P_NZPP1` a `P_NZPP3`, celdas no vacías de tres planos de 0,125 s),
  `sigma_y0_f`, `sigma_y1_f` (`P_sigma_y0`, `P_sigma_y1`, dispersión sobre los ejes rotados 45°) y `sigma_ed_f`
  (`P_sigma_ED`, distancia euclídea entre puntos sucesivos).

**Contigüidad:** una diferencia sucesiva, un par de Poincaré, un trío o una plantilla armados con dos NN que **no son
vecinos** (entre ellos falta un intervalo) inventan un intervalo que no existe. Por eso la
ventana se parte en tramos donde `contiguo` es falso y todo lo que mira más de un NN se arma **dentro** de cada tramo;
recién después se juntan los de todos los tramos. Lo que mira cada NN por separado (CV, histograma de ShEn, media m_x)
usa **todos** los NN de la ventana, vecinos o no. Los ayudantes (`_como_arreglos`, `_tramos`, `_deltas`,
`_shannon_bits` y los de Poincaré) son privados y sin el sufijo `_f`.

**Unidades:** las cuentas van en segundos (x = `nn_ms` / 1000), salvo `rmssd_f`, que devuelve ms. Las comparaciones contra
un umbral o una celda (|Δ| > 50 ms, |x_i - x_j| <= 50 ms, ⌊ms / 125⌋) se hacen **en ms**: en segundos, 1,001 - 0,751 da
0,2499999… y caería en la celda de al lado. Los cocientes de dos cantidades en la misma unidad (CV, NMASD) dan lo mismo
en ms que en s.

**NaN:** cada función devuelve NaN si no tiene con qué calcular (ningún Δ, ningún par de NN vecinos, ningún trío,
ninguna palabra, ninguna plantilla que coincida en `samen_f`), nunca 0 ni ±inf inventados. Lo que sí se calcula, aunque
dé cero, es un valor: `ssamen_f` da 0.0 cuando hay pares pero ninguno coincide, y NaN sólo con menos de 2 NN. `sigma_y0_f`
y `sigma_y1_f` dan NaN con menos de 2 pares de vecinos: el desvío de un solo punto no se puede estimar y un 0.0 haría
parecer perfectamente regular a una ventana con un solo par de vecinos. `cv_f` da 0.0 con un solo NN (el desvío de un
valor es 0), y las medias (`sigma_ed_f`, `rmssd_f`, NMASD…) se calculan con un solo término. El mínimo de NN por
ventana lo decide quien llama, no cada función ni `calcular_ventana`.

**Diferencias con las fuentes:** `P_NMASD` con la **media** de |Δ| (ec. 4.32; la tabla 1 del paper dice la suma, que
crecería con la cantidad de Δ de la ventana); `PNN50` y `PNN100` divididos por N, como la Task Force y la ec. 4.33, y con
|Δ|; SampEn de Richman–Moorman (las mismas plantillas para m y m + 1); `P_SSamEn` con todos los NN de la ventana;
RMSSD en ms; σ con ddof = 0 (y el desvío de y1 contra la media de y1); `P_IPD` con máximo o mínimo estricto y m_IP, σ_IP
sumados por tramo.
"""

from collections.abc import Callable

import numpy as np

# --- parámetros (§4.5.1) ------------------------------------------------------------------------------------------------

R_S = 0.05                      # tolerancia r de SampEn: 50 ms
BIN_S = 0.125                   # lado de la celda de los planos de Poincaré (NZPP): 125 ms, grilla alineada en 0
BINS_SHANNON = 10               # bins iguales entre el mínimo y el máximo de la ventana en ShEn

# Los mismos r y lado de celda en ms, que es donde se comparan (|x_i − x_j| ≤ r, ⌊ms / lado⌋): en s, 1,001 − 0,751 da
# 0,2499999… y caería en la celda de al lado. Se pasan a ms una sola vez, acá, y no en cada función (`csamen_f` toma su
# r de `_R_MS` también, para no mezclar dos). `round(…, 6)` saca el ruido de punto flotante del producto sin truncar un
# r que no sea de ms enteros (0,0505 s son 50,5 ms, no 50). El ruido de los NN mismos lo saca `_como_arreglos`.
_R_MS = round(R_S * 1000, 6)
_BIN_MS = round(BIN_S * 1000, 6)


# --- tramos de vecinos --------------------------------------------------------------------------------------------------


def _como_arreglos(nn_ms, contiguo) -> tuple[np.ndarray, np.ndarray]:
    """`(nn_ms, contiguo)` como `np.ndarray` (float64 y bool). Los dos tienen que tener el mismo largo: si no, algún NN
    quedaría emparejado con la contigüidad de otro.

    **Los ms se redondean a 6 decimales (1 ns)**, muy por debajo de una muestra (5 ms a 200 Hz). Los NN salen de
    `muestras / fs * 1000` y esa cuenta puede traer ruido de punto flotante (500 muestras a 800/3 Hz dan 1874,9999999999998
    ms y no 1875): con el ruido, un valor justo en un borde cae del lado equivocado, en la celda de al lado de los planos de
    Poincaré o en el bin de al lado del histograma de `shen_f`. Redondeado, el NN vuelve a ser el múltiplo exacto de una
    muestra y entonces también son exactas las cuentas que siguen (restas, celdas, bordes de los bins, umbrales). Lo demás
    sólo pierde el ruido. Todas las funciones pasan por acá (directamente o por los ayudantes de tramos, Δ y pares)."""
    nn_ms = np.round(np.asarray(nn_ms, dtype=np.float64), 6)
    contiguo = np.asarray(contiguo, dtype=bool)
    if nn_ms.ndim != 1 or nn_ms.shape != contiguo.shape:
        raise ValueError(f"nn_ms y contiguo tienen que ser vectores del mismo largo ({nn_ms.shape} y {contiguo.shape})")
    return nn_ms, contiguo


def _tramos(nn_ms, contiguo) -> list[np.ndarray]:
    """Los NN de la ventana partidos en tramos de vecinos: un corte antes de cada NN con `contiguo` falso (el primero
    abre el primer tramo, diga lo que diga su `contiguo`). Dentro de un tramo, todo NN empieza donde terminó el
    anterior; entre tramos, no. Una ventana sin NN da un solo tramo vacío."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    cortes = np.flatnonzero(~contiguo)
    return np.split(nn_ms, cortes[cortes > 0])


def _deltas(nn_ms, contiguo) -> np.ndarray:
    """Los Δ(n) = x(n+1) − x(n) **en ms** entre vecinos, de todos los tramos juntos y en el orden del tiempo. Un NN
    cuyo `contiguo` es falso no tiene Δ con el anterior: el salto sobre un latido que salió del NN no es una
    diferencia sucesiva."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    return np.diff(nn_ms)[contiguo[1:]]


# --- dispersión ---------------------------------------------------------------------------------------------------------


def cv_f(nn_ms, contiguo) -> float:
    """`P_cv`: coeficiente de variación, σ_x / m_x (ec. 4.30), con σ_x el desvío (ddof = 0) y m_x la media de **todos**
    los NN de la ventana: no mira vecinos, así que `contiguo` no cambia nada. Es un cociente de dos cantidades en la misma
    unidad: da lo mismo en ms que en s. NaN si no hay NN."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    if nn_ms.size == 0:
        return float("nan")
    return float(np.std(nn_ms) / np.mean(nn_ms))


def rmssd_f(nn_ms, contiguo) -> float:
    """`P_RMSSD`: raíz de la media de los Δ² entre vecinos (ec. 4.31), **en ms** (las demás van en s; el RMSSD clínico se
    lee en ms). NaN si no hay ningún Δ."""
    delta = _deltas(nn_ms, contiguo)
    if delta.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(delta ** 2)))


def nmasd_f(nn_ms, contiguo) -> float:
    """`P_NMASD`: media de |Δ| entre vecinos sobre m_x, la media de todos los NN (ec. 4.32). Cociente de dos cantidades
    en la misma unidad: da lo mismo en ms que en s.

    **Es la media, no la suma.** La tabla 1 del paper dice «sum(abs(diff(series))) / mean(series)»; con la suma el valor
    crecería con la cantidad de Δ de la ventana (con la FC y con los cortes de `contiguo`). NaN si no hay ningún Δ."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    delta = _deltas(nn_ms, contiguo)
    if delta.size == 0:
        return float("nan")
    return float(np.mean(np.abs(delta)) / np.mean(nn_ms))


def ipd_f(nn_ms, contiguo) -> float:
    """`P_IPD`: índice de puntos de inflexión, (N_IP − m_IP) / σ_IP (ec. 4.34 a 4.36).

    N_IP cuenta los NN que son máximo o mínimo local **estricto** de su trío de vecinos: un empate (x(n) = x(n−1), por
    ejemplo) no es punto de inflexión. Para una serie al azar de N_s NN seguidos, el número esperado de puntos es
    2(N_s − 2)/3 y su varianza (16 N_s − 29)/90. Con tramos de vecinos, esos dos se **suman por tramo** de largo
    N_s ≥ 3 (un tramo más corto no tiene tríos): m_IP y σ_IP² son esas sumas. NaN si ningún tramo tiene 3 NN."""
    n_ip = 0
    esperado = 0.0
    varianza = 0.0
    hay_trio = False
    for tramo in _tramos(nn_ms, contiguo):
        n = len(tramo)
        if n < 3:
            continue
        hay_trio = True
        previo, centro, siguiente = tramo[:-2], tramo[1:-1], tramo[2:]
        n_ip += np.count_nonzero(((centro > previo) & (centro > siguiente))
                                 | ((centro < previo) & (centro < siguiente)))
        esperado += 2 * (n - 2) / 3
        varianza += (16 * n - 29) / 90
    if not hay_trio:
        return float("nan")
    return float((n_ip - esperado) / np.sqrt(varianza))


def _pnn(nn_ms, contiguo, k_ms: float) -> float:
    """100 · #(|Δ| > k) / N, en %: los Δ entre vecinos que pasan `k_ms` sobre **todos** los NN de la ventana (N), como la
    Task Force 1996 (NN50 / el total de intervalos NN) y la ec. 4.33, pero **con valor absoluto** (la ecuación cuenta
    Δ > k sin él, o sea sólo los RR que se alargan; la Task Force y la tabla 1 del paper dicen «differing», en los dos
    sentidos). Estricto: un Δ de justo k no cuenta. NaN si no hay ningún Δ (nada que comparar), no 0 %."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    delta = _deltas(nn_ms, contiguo)
    if delta.size == 0:
        return float("nan")
    return float(100 * np.count_nonzero(np.abs(delta) > k_ms) / nn_ms.size)


def pnn50_f(nn_ms, contiguo) -> float:
    """`PNN50`: 100 · #(|Δ| > 50 ms) / N, en % (ec. 4.33, con valor absoluto; ver `_pnn`). Se divide por N y no por los
    Δ entre vecinos, como la Task Force y el PFC."""
    return _pnn(nn_ms, contiguo, 50.0)


def pnn100_f(nn_ms, contiguo) -> float:
    """`PNN100`: lo mismo que `pnn50_f` con 100 ms: 100 · #(|Δ| > 100 ms) / N, en %."""
    return _pnn(nn_ms, contiguo, 100.0)


# --- entropía y regularidad ---------------------------------------------------------------------------------------------


def _shannon_bits(cuentas: np.ndarray) -> float:
    """−Σ p · log₂ p, en bits, de un vector de cuentas (las celdas vacías no suman). Si hay una sola celda con
    cuentas, 0.0 (y no −0.0: `log2(1 / p)` con p = 1 es 0). Lo comparten `shen_f` (cuentas de un histograma) y
    `shan2_f` (cuentas de palabras); cada una arma sus cuentas y dice qué son."""
    cuentas = np.asarray(cuentas, dtype=np.float64)
    p = cuentas[cuentas > 0] / cuentas.sum()
    return float(np.sum(p * np.log2(1 / p)))


def samen_f(nn_ms, contiguo) -> float:
    """`P_SamEn`: entropía muestral con m = 1 y r = `R_S` = 0,05 s, SampEn = −ln(A / B) (ec. 4.39 a 4.41, con la forma de
    Richman y Moorman).

    **Plantillas:** los NN `i` cuyo siguiente `i + 1` es vecino (`contiguo[i + 1]`). Son las mismas para m = 1 y m = 2
    (esa es la forma de Richman–Moorman: B y A se cuentan sobre las mismas plantillas, así el cociente es una
    probabilidad condicional), y por eso una plantilla sin siguiente vecino queda fuera: compararla en m = 2 sería armar
    un par (x(i), x(i + 1)) con dos NN que no son vecinos. **B**: los pares (i, j), i ≠ j, de plantillas con
    |x_i − x_j| ≤ r; **A**: de esos, los que además cumplen |x_{i+1} − x_{j+1}| ≤ r. Los pares se cuentan sin orden (el
    par (i, j) y el (j, i) son el mismo): el cociente A / B es el mismo que con orden. La comparación es en ms (≤ 50),
    no en s, para que un Δ de justo 50 ms no se pierda por redondeo.

    **NaN si B = 0 o A = 0:** sin coincidencias de 1 NN no hay con qué comparar, y con A = 0 el logaritmo sería +inf
    (la entropía muestral no está definida: ninguna plantilla que coincidió en 1 siguió coincidiendo en 2). Es O(n²) por
    ventana, que con n ≈ 35 no pesa. Los índices y la normalización de las ec. 4.40–4.41 del PFC no coinciden entre
    sí; acá se usa la forma de Richman y Moorman de arriba."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    plantilla = np.flatnonzero(contiguo[1:])            # i tal que el NN i + 1 es vecino del i
    x, siguiente = nn_ms[plantilla], nn_ms[plantilla + 1]
    coincide_x = np.abs(x[:, None] - x[None, :]) <= _R_MS
    coincide_siguiente = np.abs(siguiente[:, None] - siguiente[None, :]) <= _R_MS
    b = np.count_nonzero(np.triu(coincide_x, k=1))                          # pares i < j (sin orden, sin la diagonal)
    a = np.count_nonzero(np.triu(coincide_x & coincide_siguiente, k=1))
    if a == 0 or b == 0:
        return float("nan")
    return float(-np.log(a / b))


def ssamen_f(nn_ms, contiguo) -> float:
    """`P_SSamEn`: B(1, r) / m_x (ec. 4.43), con r = `R_S` = 0,05 s, m_x la media de **todos** los NN en s y B(1, r) la
    fracción de pares (i, j), i ≠ j, de **todos** los NN de la ventana con |x_i − x_j| ≤ r, sobre los n(n − 1)/2 pares
    posibles. Con m = 1 la plantilla es un solo NN y no necesita vecino: `contiguo` no cambia nada. La comparación es en
    ms (≤ 50). B es una fracción sin unidad, así que dividirla por m_x en ms en vez de en s cambiaría el valor por 1000:
    m_x va en s. NaN con menos de 2 NN (no hay ningún par); si hay pares pero ninguno coincide, 0.0: la fracción existe y
    vale 0."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    n = nn_ms.size
    if n < 2:
        return float("nan")
    coincide = np.abs(nn_ms[:, None] - nn_ms[None, :]) <= _R_MS
    fraccion = np.count_nonzero(np.triu(coincide, k=1)) / (n * (n - 1) / 2)
    return float(fraccion / (np.mean(nn_ms) / 1000))


def csamen_f(nn_ms, contiguo) -> float:
    """`P_CSamEn`: entropía muestral corregida, `samen_f` + ln(2 · `R_S`) − ln(m_x) con m_x la media de **todos** los NN
    de la ventana **en s** (ec. 4.42). Suma ln(2r), que le saca a la SampEn lo que le pone la tolerancia, y resta ln(m_x),
    que la lleva a la escala de los NN de la ventana. No es un cociente de unidades iguales: m_x y 2r tienen que ir los
    dos en s. El r es `_R_MS` / 1000, el mismo que usa `samen_f`, y no `R_S` leído aparte, para no mezclar dos r. NaN si
    `samen_f` lo es (A = 0 o B = 0)."""
    samen = samen_f(nn_ms, contiguo)
    if np.isnan(samen):
        return float("nan")
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    return float(samen + np.log(2 * _R_MS / 1000) - np.log(np.mean(nn_ms) / 1000))  # el mismo r que usó `samen_f`


def shen_f(nn_ms, contiguo) -> float:
    """`P_ShEn`: entropía de Shannon, −Σ p · log₂ p, en bits (ec. 4.37 y 4.38), del histograma de **todos** los NN de la
    ventana con `BINS_SHANNON` = 10 bins iguales entre el mínimo y el máximo de la ventana (el último bin es cerrado: el
    máximo cuenta). No mira vecinos: `contiguo` no cambia nada. Mide la forma de la distribución de los valores, no su
    ancho (los bins se estiran al rango de cada ventana). Si todos los NN son iguales (rango 0), 0.0. NaN sin NN."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    if nn_ms.size == 0:
        return float("nan")
    if nn_ms.max() == nn_ms.min():
        return 0.0
    cuentas, _ = np.histogram(nn_ms, bins=BINS_SHANNON)
    return _shannon_bits(cuentas)


def shan2_f(nn_ms, contiguo) -> float:
    """`P_Shan2`: **definición propia**. El paper (tabla 1) la describe como «possibly over binarized or discretized NN
    series» y no la define; el PFC tampoco. Lo que sigue es lo que se hace acá:

    - Cada Δ entre vecinos se binariza: 1 si Δ > 0 (el RR se alarga), 0 si no (**incluye Δ = 0**).
    - Palabras de 3 símbolos seguidos **dentro de un tramo** de vecinos (cuatro NN vecinos): 8 posibles. Una palabra
      que cruza un corte uniría Δ que no existen, así que no se arma.
    - `P_Shan2` = −Σ p · log₂ p de la frecuencia de las palabras: entre 0 y 3 bits. NaN si no hay ninguna palabra
      (ningún tramo de 4 NN).

    Mide el orden de los latidos, que `shen_f` (la distribución de valores) no ve. Una serie al azar da ~2,8 bits (no
    3: los ocho patrones de subidas y bajadas de cuatro valores iid no son equiprobables: 1, 3, 5, 3, 3, 5, 3, 1 casos
    de 24); un sinusal modulado por la respiración, con rachas de subidas y bajadas, menos. Como Δ = 0 cuenta como 0
    y los RR están cuantizados a 1/fs, el sinusal muy regular da muchos «000»."""
    palabras = []
    for tramo in _tramos(nn_ms, contiguo):
        if len(tramo) < 4:
            continue
        sube = (np.diff(tramo) > 0).astype(np.int64)         # un bit por Δ entre vecinos
        palabras.append(4 * sube[:-2] + 2 * sube[1:-1] + sube[2:])    # la palabra de 3 bits como un entero 0–7
    if not palabras:
        return float("nan")
    return _shannon_bits(np.bincount(np.concatenate(palabras), minlength=8))


# --- Poincaré -----------------------------------------------------------------------------------------------------------


def _pares_de_nn(nn_ms, contiguo) -> tuple[np.ndarray, np.ndarray]:
    """Los pares (x(n), x(n + 1)) **en ms** entre vecinos, de todos los tramos juntos y en el orden del tiempo, como dos
    arreglos del mismo largo: el NN y su siguiente. El par que cruza un corte (el NN siguiente no es vecino) no existe. Sus
    diferencias son los `_deltas`."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    vecino = contiguo[1:]                      # vecino[n] dice si el NN n + 1 empieza donde terminó el n
    return nn_ms[:-1][vecino], nn_ms[1:][vecino]


def _pares_de_deltas(nn_ms, contiguo) -> tuple[np.ndarray, np.ndarray]:
    """Los pares (Δ(n), Δ(n + 1)) **en ms**, de tres NN vecinos seguidos (x(n), x(n + 1), x(n + 2)), de todos los tramos
    juntos, como dos arreglos del mismo largo. Un par de Δ que cruza un corte uniría dos diferencias que no comparten NN,
    así que no se arma; con menos de 3 NN seguidos en un tramo no hay ninguno."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    delta = np.diff(nn_ms)                     # delta[n] = Δ(n); sólo vale si contiguo[n + 1]
    vecino = contiguo[1:]
    tres_vecinos = vecino[:-1] & vecino[1:]    # Δ(n) y Δ(n + 1) valen los dos
    return delta[:-1][tres_vecinos], delta[1:][tres_vecinos]


def _celdas(valor_ms) -> np.ndarray:
    """Índice de la celda de la grilla de `BIN_S` = 0,125 s, alineada en 0, de cada valor dado **en ms**: ⌊ms / 125⌋ (el
    piso, no el truncado: −50 ms es la celda −1, no la 0). En ms y no en s, porque 1,001 − 0,751 da 0,2499999… y caería en
    la celda de al lado."""
    return np.floor(np.asarray(valor_ms, dtype=np.float64) / _BIN_MS).astype(np.int64)


def _cuenta_distintas(columna: np.ndarray, fila: np.ndarray) -> float:
    """Cuántas celdas distintas (columna, fila) hay entre los puntos dados por sus índices de celda: las no vacías, como
    `float`, que es lo que devuelven las 17."""
    return float(len(set(zip(columna.tolist(), fila.tolist()))))


def nzpp1_f(nn_ms, contiguo) -> float:
    """`P_NZPP1`: celdas no vacías (NZPP, «non-zero Poincaré plane») del plano (x(n), x(n + 1)), el de los NN sucesivos,
    con celdas de `BIN_S` = 0,125 s alineadas en 0 (§4.5.1). Un punto por par de NN **vecinos**: el par que cruza un
    corte no es un punto del plano. Cuenta celdas, no puntos: una ventana de NN todos iguales es una sola celda. NaN si
    no hay ningún par de vecinos."""
    x, siguiente = _pares_de_nn(nn_ms, contiguo)
    if x.size == 0:
        return float("nan")
    return _cuenta_distintas(_celdas(x), _celdas(siguiente))


def nzpp2_f(nn_ms, contiguo) -> float:
    """`P_NZPP2`: celdas no vacías del plano (Δ(n), Δ(n + 1)), el de las diferencias sucesivas, **sin las cuatro
    centrales**: las que tocan el origen, con los dos índices en {−1, 0} (−125 ≤ Δ < 125 ms en cada eje). Un punto por trío de NN
    vecinos. En una serie regular casi todos los puntos caen en esas cuatro; esta cuenta mide cuánto se sale la ventana de
    ahí, en cuántas celdas distintas. NaN si no hay ningún trío de vecinos; **0** si hay puntos pero todos caen en el
    centro (el valor existe y vale 0)."""
    delta, delta_siguiente = _pares_de_deltas(nn_ms, contiguo)
    if delta.size == 0:
        return float("nan")
    columna, fila = _celdas(delta), _celdas(delta_siguiente)
    fuera = (columna < -1) | (columna > 0) | (fila < -1) | (fila > 0)      # algún índice fuera de {−1, 0}
    return _cuenta_distintas(columna[fuera], fila[fuera])


def nzpp3_f(nn_ms, contiguo) -> float:
    """`P_NZPP3`: celdas no vacías del plano (x(n), Δ(n)), el NN contra su diferencia con el siguiente, con celdas de
    `BIN_S` = 0,125 s alineadas en 0 en los dos ejes (el Δ negativo cae en celdas negativas: ⌊−50 / 125⌋ = −1). Un punto
    por par de NN **vecinos**. NaN si no hay ningún par de vecinos."""
    x, siguiente = _pares_de_nn(nn_ms, contiguo)
    if x.size == 0:
        return float("nan")
    return _cuenta_distintas(_celdas(x), _celdas(siguiente - x))


def sigma_y0_f(nn_ms, contiguo) -> float:
    """`P_sigma_y0`: σ(y0) / m_x (ec. 4.46, 4.47 y 4.50), con y0 = Δ(n) / √2 la coordenada del plano de Poincaré sobre el
    eje perpendicular a la diagonal (los ejes rotados 45°): mide la dispersión de las diferencias sucesivas, el ancho de
    la nube (la variabilidad latido a latido). σ con ddof = 0 sobre los pares de NN **vecinos**; m_x, la media de **todos**
    los NN de la ventana, la lleva a un cociente sin unidad (los dos en la misma unidad: da lo mismo en ms que en s). **NaN
    con menos de 2 pares de vecinos**: el desvío de un solo punto no es estimable, y un 0.0 haría parecer «perfectamente
    regular» a una ventana con un solo par de vecinos."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    delta = _deltas(nn_ms, contiguo)
    if delta.size < 2:
        return float("nan")
    y0 = delta / np.sqrt(2)
    return float(np.std(y0) / np.mean(nn_ms))


def sigma_y1_f(nn_ms, contiguo) -> float:
    """`P_sigma_y1`: σ(y1) / m_x (ec. 4.45 y 4.47 con j = 1, normalizada como la 4.50), con y1 = (x(n) + x(n + 1)) / √2
    la coordenada sobre la diagonal (los ejes rotados 45°): mide cuánto se mueve el nivel medio de los latidos a lo largo
    de la ventana, el largo de la nube. σ con ddof = 0 **contra la media de y1** (la ec. 4.47 escribe la media de y0 para
    las dos; acá, la de y1, porque y0 promedia casi 0 y y1, √2 veces el NN medio) sobre los pares de NN **vecinos**;
    m_x, la media de **todos** los NN de la ventana. **NaN con menos de 2 pares de vecinos**, igual que `sigma_y0_f` (el
    desvío de un solo punto no es estimable y un 0.0 se leería como una ventana perfectamente regular)."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    x, siguiente = _pares_de_nn(nn_ms, contiguo)
    if x.size < 2:
        return float("nan")
    y1 = (x + siguiente) / np.sqrt(2)
    return float(np.std(y1) / np.mean(nn_ms))


def sigma_ed_f(nn_ms, contiguo) -> float:
    """`P_sigma_ED`: media de la distancia euclídea entre puntos sucesivos del plano de Poincaré, sobre m_x (ec. 4.49 y
    4.51). El punto n es (x(n), x(n + 1)) y el n + 1, (x(n + 1), x(n + 2)); su distancia es √(Δ(n)² + Δ(n + 1)²): cuánto
    salta la nube de un latido al siguiente. Un valor por trío de NN **vecinos**; m_x, la media de **todos** los NN de la
    ventana. NaN si no hay ningún trío de vecinos."""
    nn_ms, _ = _como_arreglos(nn_ms, contiguo)
    delta, delta_siguiente = _pares_de_deltas(nn_ms, contiguo)
    if delta.size == 0:
        return float("nan")
    return float(np.mean(np.sqrt(delta ** 2 + delta_siguiente ** 2)) / np.mean(nn_ms))


# --- el registro y la ventana entera ------------------------------------------------------------------------------------

# Nombre de columna → función, en el orden de la tabla 1 del paper. `calcular_ventana` lo recorre y nada más: una
# característica que no esté acá no sale.
CARACTERISTICAS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "P_cv": cv_f,
    "P_RMSSD": rmssd_f,
    "P_NMASD": nmasd_f,
    "P_IPD": ipd_f,
    "PNN50": pnn50_f,
    "PNN100": pnn100_f,
    "P_SamEn": samen_f,
    "P_SSamEn": ssamen_f,
    "P_CSamEn": csamen_f,
    "P_ShEn": shen_f,
    "P_Shan2": shan2_f,
    "P_NZPP1": nzpp1_f,
    "P_NZPP2": nzpp2_f,
    "P_NZPP3": nzpp3_f,
    "P_sigma_y0": sigma_y0_f,
    "P_sigma_y1": sigma_y1_f,
    "P_sigma_ED": sigma_ed_f,
}


def calcular_ventana(nn_ms, contiguo) -> dict[str, float]:
    """Las 17 características de una ventana: `{columna: valor}` en el orden de `CARACTERISTICAS`, con NaN en las que no
    se pueden calcular. **No mira el mínimo de NN**: decidir si la ventana tiene NN suficientes es de quien la arma, que
    no llama acá con las que no los tienen. Pasa los dos vectores a `np.ndarray` una vez (float64 y bool) y revisa que
    tengan el mismo largo, antes de darle a cada función lo que ella espera."""
    nn_ms, contiguo = _como_arreglos(nn_ms, contiguo)
    return {columna: funcion(nn_ms, contiguo) for columna, funcion in CARACTERISTICAS.items()}
