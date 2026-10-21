# Guía de instrucciones para reproducir el TP4 con IA

**Herramienta:** Claude Opus 5.5 en Claude Code (agente con acceso al repositorio y a la terminal).
**Propósito:** generar el código fuente (`src/`), armar el notebook y redactar las interpretaciones a partir de los
resultados. Las decisiones metodológicas se tomaron en cada punto de control (🔎) antes de seguir.

La guía tiene una instrucción por etapa. Cada instrucción dice **qué hay que pedir**, **qué hay que verificar** antes
de aceptar el resultado y **qué se decidió**. Para reproducir el trabajo, hay que darlas en orden en una sesión
abierta en la raíz del repositorio.

---

## 0. Contexto y estructura

> Leé la carpeta `TP3/` (README, `config.toml`, `src/`, notebook) y replicá la misma arquitectura en `TP4/`:
> `config.toml` con todos los parámetros, `src/` con un módulo por etapa del flujo de datos, `data/raw` para los
> crudos inmutables, `data/processed` para las salidas derivadas (ninguna de las dos se versiona) y `notebooks/TP4.ipynb`.
> Usá el entorno virtual `.env` del repo. El notebook tiene que poder regenerar todo desde cero.

🔎 Verificar que `.gitignore` excluya `data/` salvo los `.gitkeep` y que los nombres de módulos sigan la convención del TP3.

## 1. Ingesta y One Big Table

> Armá el manifiesto de la ventana a partir de `masterfilelist.txt`, sin bajar el archivo completo (~130 MB):
> pedí solo la cola con un HTTP Range y agrandala hasta cubrir el inicio de la ventana. Fijá la ventana en
> `config.toml` (7 días completos en UTC, del 13 al 19/10/2026) para que sea reproducible. Descargá los `export.CSV.zip`
> en paralelo con reintentos y backoff, **validando el MD5 del manifiesto**, y parseá las 61 columnas del codebook con
> pyarrow, con tipos explícitos: los códigos CAMEO como texto y sin comillas como delimitador.
> Deduplicá por `GlobalEventID` quedándote con el `DATEADDED` más reciente.

🔎 Verificar: 672 archivos esperados (7 × 96), 0 faltantes, 0 MD5 inválidos, `GlobalEventID` único.

> Construí una **tabla desnormalizada** (OBT) con las 61 columnas originales y, al lado, las descripciones de todos los
> códigos (CAMEO evento/base/raíz, QuadClass y sus ejes, tipos y países de los actores, países FIPS de las tres
> geografías, tipo de geocodificación) y atributos derivados: fecha del evento, fecha de publicación, desfase en días,
> hora, día de la semana y dominio del medio. **Conservá los nulos como vienen**: se tratan en la limpieza.
> Guardala en parquet.

🔎 Revisar el diccionario de columnas derivadas y una muestra de filas. Decisión: la OBT reemplaza al esquema estrella
del TP3 porque el análisis no necesita JOINs.

## 2. Diagnóstico de nulos y ceros

> Perfilá cada columna del export: % de nulos y, en las numéricas, % de ceros. Antes de decidir nada, separá
> **nulos estructurales** (la ausencia significa algo), **nulos reales** (el dato debería existir), **ceros
> estructurales** (un 0 que codifica "sin dato") y **ceros válidos** (un 0 que es un valor de la escala), según el
> codebook. Mostrame la evidencia de cada caso (qué códigos tienen Goldstein nulo, qué acciones tienen Goldstein 0,
> dónde está la latitud 0, qué valores toma el desfase).

🔎 Puntos de control y decisiones del equipo:
- Actor ausente o sin tipo → categoría explícita `SIN_ACTOR` / `SIN_TIPO`, **no** descartar filas (se perdería el 30 %
  de los eventos y se sesgaría la muestra hacia eventos con dos actores).
- `ActionGeo_Type = 0` → cero estructural → `SIN_UBICACION`.
- Goldstein nulo (3 eventos, código `1213` fuera del codebook) → imputar con la tabla `CAMEO.goldsteinscale.txt` por
  jerarquía código → base → raíz. Pedir que se agregue esa tabla a las *lookups*.
- Columnas con > 90 % de nulos → fuera del análisis (pero siguen en la OBT). Se eligió 90 % y no 60 % porque con 60 % se
  perdía `Actor2Type1Code`, cuyos nulos son estructurales.

> Agregá un chequeo de *near zero variance* (coeficiente de variación y razón entre la moda y el segundo valor) y la
> correlación entre los conteos en escala log.

🔎 Decisión: `NumSources` y `LagDays` serían NZV por razón de frecuencias, pero **se conservan** porque su rareza es la
señal que se busca. `NumArticles` sale por redundancia (r = 0,99 con `NumMentions`).

## 3. Matriz de datos

> Reutilizá del TP3 la reducción de cardinalidad (`OTHER` para frecuencias < 5 %) y la codificación en indicadoras con
> una categoría de referencia por variable. Categóricas: raíz CAMEO, tipo de actor 1 y 2, tipo de geocodificación y país
> de la acción. Continuas: Goldstein, tono y `log1p` de menciones, fuentes y desfase. Más `IsRootEvent`.

🔎 En el primer ACP apareció un autovalor exactamente 0. Pedir: *"identificá qué combinación lineal genera el
autovalor nulo mirando el último autovector"*. Resultado: las indicadoras de `QuadClass` son combinación lineal exacta
de las de la raíz CAMEO (todas las raíces 01–05 superan el 5 % y la referencia de `QuadClass` es Cooperación verbal).
Decisión: sacar `QuadClass` de la matriz, dejarla solo para interpretar y agregar al notebook una celda que **demuestre**
la dependencia.

## 4. ACP

> Escalá con `StandardScaler` y ajustá el ACP completo. Para la cantidad mínima de componentes compará Kaiser, varianza
> acumulada ≥ 80 %, Jolliffe y **análisis paralelo de Horn** (permutando cada columna por separado sobre una muestra
> de 50 000 filas, 20 repeticiones, P95), y mostralos en el gráfico de sedimentación.

🔎 Verificar media ≈ 0 y desvío = 1 de la matriz escalada, y que la varianza total sea igual a p. Decisión: **K = 15**
(Horn = Kaiser), ya que no hay codo claro y Horn es el único criterio que compara contra ruido.

> Calculá las cargas (autovector × √autovalor), el mapa de calor y las 4 variables de mayor |carga| por componente.

🔎 Revisar que la lectura de cada componente esté respaldada por las cargas (PC1 geografía, PC2 cooperación/conflicto…).

## 5. Atípicos multivariantes

> Sobre el modelo de K componentes calculá el **T² de Hotelling** (límite χ²_K) y el **SPE/Q** (límite de
> Jackson–Mudholkar) con α = 0,001. Marcá como atípico lo que supere cualquiera de los dos y clasificalo como
> `solo T2`, `solo Q` o `T2 y Q`. Para cada atípico calculá las **contribuciones por variable** a T² y a Q, y asignale un
> motivo según la variable que más contribuye. Hacé el gráfico de control T² vs Q.

🔎 Verificar que las contribuciones a T² de cada fila sumen su T². Discutir por qué hay 11 veces más atípicos de los
esperados bajo normalidad (indicadoras binarias y colas pesadas). Decisión: mantener α = 0,001 y **priorizar por puntaje**.

> Perfilá los atípicos frente al resto (medianas, *lift* por categoría), agrupalos por `SOURCEURL` y mostrá ejemplos
> de cada motivo. Exportá la tabla de atípicos aparte, en parquet y CSV.

🔎 Abrir las URLs de los casos de mayor puntaje para confirmar la lectura (sindicación con el mismo tono en todas las
fuentes; desfases de 3650 días en noticias actuales).

## 6. Redacción

> Con los resultados ya ejecutados, redactá las interpretaciones de cada sección y las conclusiones citando solo
> números que aparezcan en las salidas del notebook. Después ejecutalo de punta a punta y verificá que no haya errores
> ni advertencias.

🔎 Revisión final del equipo: contrastar cada cifra del texto con su celda de salida.
