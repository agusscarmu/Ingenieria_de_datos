# TP1 — Análisis de eventos globales con GDELT 2.0

**Materia:** Ingeniería de Datos

Notebook: [`TP1.ipynb`](TP1.ipynb)

Se calculan tres medidas diarias sobre los eventos registrados por
[GDELT 2.0](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
a partir del **2 de septiembre de 2026**:

- Cantidad de eventos en el día
- Promedio de artículos por evento
- Promedio de fuentes por evento

Los datos se leen de los archivos `export.CSV.zip` que GDELT publica cada 15 minutos
(96 por día), listados en `masterfilelist.txt`. El resultado se presenta como un
DataFrame de Pandas.

## Declaración de uso de Inteligencia Artificial

| Herramienta | Propósito |
|---|---|
| **Gemini 2.5 Flash** | Generación de la versión inicial del código fuente (descarga y parseo de los archivos de GDELT) y redacción preliminar de los textos. |
| **Claude Opus 5** | Revisión y corrección del código: detección de un error en el mapeo de columnas del `export.CSV`, extensión del cálculo a la serie diaria completa, optimización del acceso a los datos (descarga paralela y agregación incremental) y revisión de la redacción. |

Las decisiones metodológicas, la validación de los resultados contra el *codebook*
oficial de GDELT y la interpretación final son de elaboración propia.

## Ejecución

```bash
pip install pandas numpy requests matplotlib
jupyter notebook TP1.ipynb
```

El notebook analiza el rango **2/09/2026 – 10/09/2026**. Las fechas están fijadas de forma
explícita para que el resultado sea reproducible: reejecutarlo en otra fecha devuelve la misma
tabla. El 10/09 es el día en que se corrió el análisis, por lo que está incompleto (89 de 96
archivos); el notebook lo identifica en la columna `dia_completo` de la tabla de control.
