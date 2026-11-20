"""Almacén (SQLite, esquema estrella): se crea desde sql/almacen.sql y se carga leyendo sólo la base transaccional (ETL 2).

`construir` recorre los Holter de la transaccional y, de cada ventana de 30 s entera, decide si es un hecho:

- **RR de la ventana:** salen de las posiciones de los latidos (`ventanas.rr_de_ventanas`), tal como vienen.
- **Ritmo de la ventana:** cada latido toma el ritmo del último episodio que empezó en él o antes. Si todos los latidos
  de la ventana son `AFIB`, la ventana es `AFIB`; si todos son `N`, es `N`; si no (mezcla, otro ritmo, o ningún
  latido), no tiene ritmo. Un Holter sin anotar tiene todas sus ventanas en `N`.
- **Precedencia:** una ventana con latidos y sin ritmo se descarta por ritmo; si no, con menos de `MIN_RR` RR (incluida
  la que no tiene latidos) se descarta por pocos RR; el resto son hechos, con sus 17 características.
- **Nulos:** una característica que no se puede calcular (NaN) queda NULL en el hecho, sin imputar.

El tiempo no lleva fecha, sólo la hora del día: `tiempo_key` es la ventana del día (de 0 a 2.879) en que empieza la
ventana del Holter, y vuelve a 0 a la medianoche, así que un Holter de más de 24 h reutiliza las mismas claves.
"""
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

import caracteristicas
import config
import ventanas

SEGUNDOS_POR_DIA = 24 * 3600
DIM_RITMO = [(1, "AFIB", "FA"), (2, "N", "sin FA (N)")]          # filas de dim_ritmo: (ritmo_key, codigo, etiqueta)
CLAVE_RITMO = {codigo: clave for clave, codigo, _ in DIM_RITMO}

COLUMNAS_HECHO = ["holter_key", "n_ventana", "tiempo_key", "ritmo_key", "n_rr", *caracteristicas.CARACTERISTICAS]
INSERTAR_HECHO = (f"INSERT INTO hecho_ventana ({', '.join(COLUMNAS_HECHO)}) "
                  f"VALUES ({', '.join('?' * len(COLUMNAS_HECHO))})")


@dataclass
class Conteos:
    """Lo que `construir` cargó y descartó, para contarlo en el cuaderno (no se guarda en el almacén).

    `ventanas` son todas las ventanas enteras de todos los Holter: `descartadas_ritmo` (mezcla de ritmos, otro ritmo) +
    `descartadas_pocos_rr` (menos de `MIN_RR` RR, incluidas las que no tienen latidos) + `hechos`. `nulos` dice, por
    cada una de las 17 características, en cuántos hechos quedó NULL.
    """

    ventanas: int
    descartadas_ritmo: int
    descartadas_pocos_rr: int
    hechos: int
    nulos: dict[str, int]


def tiempo_key(hora_inicio: str, n_ventana: int) -> int:
    """Clave de `dim_tiempo` de la ventana `n_ventana` de un Holter que empezó a `hora_inicio` ("HH:MM", 24 h).

    Es la ventana del día en que cae (segundos desde la medianoche / 30). Pasada la medianoche, la hora del día vuelve
    a 0: por eso el módulo de 24 h.
    """
    horas, minutos = hora_inicio.split(":")
    segundos = int(horas) * 3600 + int(minutos) * 60 + config.VENTANA_S * n_ventana
    return segundos % SEGUNDOS_POR_DIA // config.VENTANA_S


def franja(hora: int) -> str:
    """`"noche"` de las 23 a las 7 (hora entera >= 23 o < 7); el resto del día es `"día"`."""
    return "noche" if hora >= config.NOCHE_DESDE or hora < config.NOCHE_HASTA else "día"


def _por_ventana(marca: np.ndarray, cortes: np.ndarray) -> np.ndarray:
    """Cuántos latidos con `marca` (un bool por latido) tiene cada ventana, con la suma acumulada de las marcas."""
    acumulado = np.concatenate(([0], np.cumsum(marca)))
    return acumulado[cortes[1:]] - acumulado[cortes[:-1]]


def ritmo_ventanas(muestras: np.ndarray, fs: int, n_muestras: int, episodios: list[tuple[int, str]],
                   anotado: bool) -> list[str | None]:
    """Ritmo de cada ventana entera de un Holter: `"AFIB"`, `"N"` o `None` (no se puede asignar uno).

    `muestras` son las posiciones de todos los latidos del Holter, en orden, y `episodios` los `(latido_inicio, ritmo)`
    de la transaccional. Cada latido toma el ritmo del último episodio que empezó en él o antes. La ventana es `AFIB` si
    todos sus latidos lo son, `N` si todos lo son, y `None` en cualquier otro caso, también si no tiene latidos. Con
    `anotado` falso no hay episodios y toda ventana es `N`.
    """
    if not anotado:
        return ["N"] * ventanas.n_ventanas(n_muestras, fs)
    muestras = np.asarray(muestras, dtype=np.int64)
    episodios = sorted(episodios)
    inicios = np.array([inicio for inicio, _ in episodios], dtype=np.int64)
    # Un valor por episodio y uno de más al final: el de los latidos que ningún episodio cubre (índice -1)
    es_afib = np.array([ritmo == "AFIB" for _, ritmo in episodios] + [False])
    es_n = np.array([ritmo == "N" for _, ritmo in episodios] + [False])
    episodio_del_latido = np.searchsorted(inicios, np.arange(muestras.size), side="right") - 1

    cortes = ventanas.cortes(muestras, fs, n_muestras)
    n_latidos = np.diff(cortes)
    todos_afib = (n_latidos > 0) & (_por_ventana(es_afib[episodio_del_latido], cortes) == n_latidos)
    todos_n = (n_latidos > 0) & (_por_ventana(es_n[episodio_del_latido], cortes) == n_latidos)
    return ["AFIB" if afib else "N" if n else None for afib, n in zip(todos_afib.tolist(), todos_n.tolist())]


def _abrir_solo_lectura(ruta: Path) -> sqlite3.Connection:
    """Conexión a la transaccional que no puede escribirla ni crearla si falta (`as_uri` codifica espacios y otros)."""
    return sqlite3.connect(f"{Path(ruta).resolve().as_uri()}?mode=ro", uri=True)


def _crear(ruta: Path) -> sqlite3.Connection:
    """Crea el almacén de cero (borra el anterior si existe) y devuelve la conexión, con las claves foráneas activas.

    Rehacerlo en cada corrida evita filas duplicadas y errores de clave al re-ejecutar el cuaderno.
    """
    ddl = (config.SQL / "almacen.sql").read_text(encoding="utf-8")
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    for sufijo in ("", "-journal", "-wal", "-shm"):     # un diario viejo sobre la base nueva la rompería
        Path(f"{ruta}{sufijo}").unlink(missing_ok=True)
    con = sqlite3.connect(ruta)
    con.execute("PRAGMA foreign_keys = ON")             # SQLite las trae apagadas, y se prenden por conexión
    con.executescript(ddl)
    return con


def _filas_tiempo():
    """Las 2.880 filas de `dim_tiempo`: (tiempo_key, hora_inicio "hh:mm:ss", hora, franja)."""
    for clave in range(SEGUNDOS_POR_DIA // config.VENTANA_S):
        hora, resto = divmod(clave * config.VENTANA_S, 3600)
        minuto, segundo = divmod(resto, 60)
        yield clave, f"{hora:02d}:{minuto:02d}:{segundo:02d}", hora, franja(hora)


def _cargar_dimensiones(fuente: sqlite3.Connection, salida: sqlite3.Connection) -> list[tuple]:
    """Llena las tres dimensiones. Devuelve lo que hace falta de cada Holter para cargar sus ventanas,
    `(holter_id, hora_inicio, fs_hz, n_muestras, anotado)`, en el orden de `holter_key` (el de `holter_id`)."""
    holters = fuente.execute("""SELECT h.holter_id, h.paciente_id, p.sexo, h.edad, h.tipo_fa, h.anotado
                                FROM holter h JOIN paciente p USING (paciente_id)
                                ORDER BY h.holter_id""").fetchall()
    with salida:
        salida.executemany("INSERT INTO dim_tiempo (tiempo_key, hora_inicio, hora, franja) VALUES (?, ?, ?, ?)",
                           _filas_tiempo())
        salida.executemany("INSERT INTO dim_ritmo (ritmo_key, codigo, etiqueta) VALUES (?, ?, ?)", DIM_RITMO)
        salida.executemany("""INSERT INTO dim_holter (holter_key, holter_id, paciente_id, sexo, edad, tipo_fa, anotado)
                              VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           [(clave, *holter) for clave, holter in enumerate(holters, start=1)])
    return fuente.execute("SELECT holter_id, hora_inicio, fs_hz, n_muestras, anotado FROM holter "
                          "ORDER BY holter_id").fetchall()


def _medidas(rr_ms: np.ndarray, nulos: dict[str, int]) -> list[float | None]:
    """Las 17 de los RR de una ventana, en el orden de `CARACTERISTICAS`, con NaN → None (NULL). Suma en `nulos` las que
    quedan nulas. Los RR de una ventana son todos vecinos (`ventanas.contiguos`)."""
    valores = caracteristicas.calcular_ventana(rr_ms, ventanas.contiguos(len(rr_ms)))
    medidas = []
    for columna in caracteristicas.CARACTERISTICAS:
        if math.isnan(valores[columna]):
            nulos[columna] += 1
            medidas.append(None)
        else:
            medidas.append(float(valores[columna]))
    return medidas


def _cargar_holter(fuente: sqlite3.Connection, salida: sqlite3.Connection, holter_key: int, holter: tuple,
                   conteos: Conteos) -> None:
    """Clasifica las ventanas de un Holter, carga los hechos en una transacción y suma lo que pasó en `conteos`."""
    holter_id, hora_inicio, fs, n_muestras, anotado = holter
    # La posición de todos los latidos del Holter, con una sola consulta
    filas = fuente.execute("SELECT muestra FROM latido WHERE holter_id = ? ORDER BY n", (holter_id,))
    muestras = np.fromiter(chain.from_iterable(filas), dtype=np.int64)
    episodios = fuente.execute("SELECT latido_inicio, ritmo FROM episodio_ritmo WHERE holter_id = ? "
                               "ORDER BY latido_inicio", (holter_id,)).fetchall()

    ritmos = ritmo_ventanas(muestras, fs, n_muestras, episodios, bool(anotado))
    n_latidos = np.diff(ventanas.cortes(muestras, fs, n_muestras)).tolist()
    hechos = []
    for n_ventana, rr_ms in ventanas.rr_de_ventanas(muestras, fs, n_muestras):
        ritmo, n_rr = ritmos[n_ventana], len(rr_ms)
        # Precedencia: sin ritmo (y con latidos) se descarta por ritmo; si no, con pocos RR; si no, es un hecho
        if ritmo is None and n_latidos[n_ventana] > 0:
            conteos.descartadas_ritmo += 1
        elif n_rr < config.MIN_RR:
            conteos.descartadas_pocos_rr += 1
        else:
            hechos.append((holter_key, n_ventana, tiempo_key(hora_inicio, n_ventana), CLAVE_RITMO[ritmo], n_rr,
                           *_medidas(rr_ms, conteos.nulos)))
    with salida:
        salida.executemany(INSERTAR_HECHO, hechos)
    conteos.ventanas += len(ritmos)
    conteos.hechos += len(hechos)


def construir(origen: Path = config.TRANSACCIONAL, destino: Path = config.ALMACEN) -> Conteos:
    """Borra y rehace el almacén `destino` con lo que hay en la base transaccional `origen`, y cuenta lo que cargó.

    Lee `origen` de sólo lectura. Cada Holter se carga en su propia transacción.
    """
    conteos = Conteos(ventanas=0, descartadas_ritmo=0, descartadas_pocos_rr=0, hechos=0,
                      nulos=dict.fromkeys(caracteristicas.CARACTERISTICAS, 0))
    with closing(_abrir_solo_lectura(origen)) as fuente, closing(_crear(destino)) as salida:
        for clave, holter in enumerate(_cargar_dimensiones(fuente, salida), start=1):
            _cargar_holter(fuente, salida, clave, holter, conteos)
    return conteos
