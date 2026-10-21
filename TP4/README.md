# TP4 — Detección de eventos atípicos en GDELT 2.0

**Materia:** Ingeniería de Datos

Notebook: [`notebooks/TP4.ipynb`](notebooks/TP4.ipynb) · Consigna: [`notebooks/consigna_tp4.ipynb`](notebooks/consigna_tp4.ipynb) · Guía de instrucciones IA: [`GUIA_IA.md`](GUIA_IA.md)

Sobre los eventos GDELT 2.0 publicados entre el **13/10/2026 y el 19/10/2026** (672 archivos `export.CSV`, ~730 000 eventos) se busca encontrar eventos "raros" que valga la pena investigar:

1. **Ingesta:** manifiesto desde `masterfilelist.txt`, descarga con verificación MD5 y **One Big Table** en parquet.
2. **Limpieza:** nulos y ceros (estructurales vs reales), reducción de cardinalidad (`OTHER` < 5 %) y variables indicadoras.
3. **ACP:** sobre la matriz escalada; cantidad mínima de componentes con Kaiser, varianza acumulada, Jolliffe y análisis paralelo de Horn.
4. **Atípicos multivariantes:** T² de Hotelling + SPE (Q) sobre el modelo del ACP, con contribuciones por variable; exportados en una tabla aparte.

## Estructura

```
TP4/
├── README.md, config.toml, requirements.txt, .gitignore   # metadatos y configuración
├── GUIA_IA.md          # guía de instrucciones para reproducir el trabajo con IA
├── data/
│   ├── raw/            # manifiesto_export.csv, export.CSV.zip (gdelt/<inicio>_<fin>/) y tablas de códigos (lookups/)
│   └── processed/      # obt_eventos, eventos_limpios, matriz, acp_scores, eventos_atipicos (parquet) + eventos_atipicos.csv
├── src/
│   ├── config.py       # lectura de config.toml
│   ├── ingesta.py      # manifiesto (HTTP Range sobre la masterfilelist), descarga con MD5, parseo pyarrow
│   ├── obt.py          # One Big Table
│   ├── limpieza.py     # perfil de nulos/ceros, descarte, imputación, near zero variance
│   ├── matriz.py       # reducción de cardinalidad, dummies, matriz X
│   ├── acp.py          # ACP, análisis paralelo de Horn, scree plot, cargas
│   ├── atipicos.py     # T² de Hotelling, SPE/Q, contribuciones, tabla de atípicos
│   └── pipeline.py     # raw -> OBT
└── notebooks/
    ├── TP4.ipynb
    └── consigna_tp4.ipynb
```

`data/` no se versiona: todo se regenera desde GDELT.

## Resultados principales

| Etapa | Resultado |
|---|---|
| Ingesta | 672/672 archivos, 0 faltantes, MD5 verificado · OBT 729 663 × 81 |
| Limpieza | 12 columnas descartadas (> 90 % nulos) · nulos estructurales recodificados (`SIN_ACTOR`, `SIN_TIPO`, `SIN_UBICACION`) · 3 Goldstein imputados por el codebook · 0 filas perdidas |
| Matriz | 729 663 × 26 (5 continuas, 1 binaria, 20 indicadoras); `QuadClass` y `NumArticles` excluidas por redundancia |
| ACP | **15 componentes** (Horn = Kaiser), 78,5 % de la varianza |
| Atípicos | **16 702 eventos** (2,3 %): desfase de fecha (51 %), cobertura sindicada (34 %), combinaciones raras / tono extremo (15 %) |

## Ejecución

```bash
pip install -r requirements.txt
python -m src.pipeline          # manifiesto + descarga + OBT (≈5 s con los datos ya descargados)
jupyter notebook notebooks/TP4.ipynb
```

El notebook corre la pipeline en su primera celda, así que alcanza con abrirlo y ejecutarlo.
La ventana, los umbrales, las variables y el nivel α se cambian en `config.toml`.

## Declaración de uso de Inteligencia Artificial

| Herramienta | Propósito |
|---|---|
| **Claude Opus 5.5** (Claude Code) | Organización del proyecto en carpetas, código fuente de la ingesta, la OBT, la limpieza, la matriz, el ACP y la detección de atípicos, armado del notebook y redacción de las interpretaciones a partir de los resultados. |

Las decisiones metodológicas fueron discutidas y revisadas por el equipo. La guía de instrucciones (human-in-the-loop) está en [`GUIA_IA.md`](GUIA_IA.md) y al final del notebook.
