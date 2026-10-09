# TP3 — Análisis multivariante del ecosistema de noticias (GDELT 2.0)

**Materia:** Ingeniería de Datos

Notebook: [`notebooks/TP3.ipynb`](notebooks/TP3.ipynb) · Consigna: [`notebooks/consigna_tp3.ipynb`](notebooks/consigna_tp3.ipynb)

Sobre los eventos GDELT 2.0 publicados el **05/10/2026** (96 archivos `export.CSV`, ~105 000 eventos) se responde:

1. **Perfil multivariante de eventos:** vector de medias global y por país.
2. **Topología de cobertura mediática:** distancias euclídea estandarizada y de Mahalanobis entre vectores de medias.
3. **Estructura de correlación:** mapas de calor global y por país.
4. **Compresión del ecosistema de noticias:** ACP, gráfico de sedimentación, cargas y biplot con outliers (T² de Hotelling).

## Estructura

```
TP3/
├── README.md, config.toml, requirements.txt, .gitignore   # metadatos y configuración
├── data/
│   ├── raw/          # export.CSV.zip de GDELT (gdelt/AAAAMMDD/) y tablas de códigos (lookups/)
│   └── processed/    # gdelt_dw.sqlite (almacén), eventos.parquet (vista), matriz.parquet (X)
├── src/
│   ├── config.py        # lectura de config.toml
│   ├── ingesta.py       # descarga a data/raw y parseo del export.CSV (61 columnas del codebook)
│   ├── almacen.py       # esquema estrella del TP2 extendido -> SQLite
│   ├── sql/dw_schema.sql
│   ├── matriz.py        # reducción de cardinalidad ('OTHER' < 5 %), dummies, matriz X
│   ├── multivariante.py # vectores de medias, distancias, correlaciones
│   ├── acp.py           # ACP, scree plot, cargas, biplot, T² de Hotelling
│   └── pipeline.py      # raw -> processed
└── notebooks/
    ├── TP3.ipynb
    └── consigna_tp3.ipynb
```

`data/` no se versiona: todo se regenera desde GDELT.

## Almacén de datos

Se parte del esquema estrella del TP2 (grano: una fila por noticia-evento) y se extiende con:

| Requisito | Implementación |
|---|---|
| Actores 1 y 2 | `dim_actor` con rol doble (`actor1_key`, `actor2_key`), incluye país y tipo de actor |
| Tipo de interacción | nueva `dim_tipo_interaccion` (QuadClass: cooperación/conflicto × verbal/material) + `dim_cameo` |
| Región | `dim_ubicacion` (país FIPS con nombre, ADM1, coordenadas) |
| Tono y Goldstein | medidas `avg_tone` y `goldstein_scale` en `fact_cobertura_evento` |

La vista `v_eventos_analisis` aplana la estrella y alimenta la matriz de datos.

## Ejecución

```bash
pip install -r requirements.txt
python -m src.pipeline          # descarga + almacén + matriz (≈10 s con los datos ya descargados)
jupyter notebook notebooks/TP3.ipynb
```

El notebook también corre la pipeline en su primera celda, así que alcanza con abrirlo y ejecutarlo.
La fecha, el umbral de `OTHER`, las variables y los parámetros del ACP se cambian en `config.toml`.

## Declaración de uso de Inteligencia Artificial

| Herramienta | Propósito |
|---|---|
| **Claude Opus 5.5** (Claude Code) | Organización del proyecto en carpetas, código fuente de la ingesta, del almacén de datos, de la matriz y de las funciones de análisis, y armado del notebook. En los ítems en los que la consigna pide un rol de tutor socrático, la IA dejó preguntas guía en lugar de la interpretación. Las preguntas fueron respondidas con ayuda de la IA y la participación y revisión del equipo. |

Las interpretaciones de esos ítems y las decisiones metodológicas son de elaboración propia.
