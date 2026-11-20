# TP final: fibrilación auricular en Holter

Trabajo final de Ingeniería de datos. Proceso completo de ETL sobre Holter de 24 h con y sin fibrilación auricular (FA): de los datos crudos a una base transaccional, un almacén dimensional y un tablero interactivo que compara FA contra sin FA y día contra noche.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agusscarmu/Ingenieria_de_datos/blob/master/TP_FINAL/notebooks/tp_final.ipynb)

## Datos

[SHDB-AF v1.0.1](https://physionet.org/content/shdb-af/1.0.1/) (PhysioNet): 128 Holter de ~24 h de 122 pacientes, 200 Hz, 2 canales, con datos clínicos y, en 98, el ritmo anotado. Licencia Open Data Commons Attribution 1.0. El cuaderno los baja solo; no hace falta subir nada.

## Cómo correrlo

Abrir el cuaderno con el botón de arriba y ejecutar todo. La primera celda clona este repositorio para poder importar el código de `src/`.

## Estructura

```
TP_FINAL/
├── notebooks/
│   └── tp_final.ipynb       el entregable: ETL, bases y tablero, documentado paso a paso
├── src/                     código que usa el cuaderno
│   ├── config.py            rutas y constantes
│   ├── extraccion.py        baja SHDB-AF y lo lee (CSV y marcas de los Holter)
│   ├── transformacion.py    limpia el CSV y saca los episodios de ritmo
│   ├── ventanas.py          ventanas de 30 s y RR de cada una
│   ├── caracteristicas.py   las 17 características de variabilidad del ritmo
│   ├── transaccional.py     crea y carga la base transaccional
│   ├── almacen.py           crea y carga el almacén dimensional
│   └── tablero.py           tablero de Dash, figura fija con las 17 y Wilcoxon de día contra noche
├── sql/
│   ├── transaccional.sql    DDL de la base transaccional (3FN)
│   └── almacen.sql          DDL del almacén (esquema estrella)
├── data/                    se genera al correr el cuaderno (no se versiona)
│   ├── raw/                 SHDB-AF tal como se baja
│   └── db/                  transaccional.sqlite y almacen.sqlite
├── presentacion/            PDF de la defensa
└── requirements.txt
```
