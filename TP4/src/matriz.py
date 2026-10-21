"""Construcción de la matriz de datos: reducción de cardinalidad, variables indicadoras
y unión con las variables continuas (mismo criterio que en el TP3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def reducir_cardinalidad(s: pd.Series, umbral: float, etiqueta: str = "OTHER") -> pd.Series:
    """Reemplaza por `etiqueta` las categorías cuya frecuencia relativa es menor a `umbral`."""
    frec = s.value_counts(normalize=True, dropna=False)
    marginales = frec.index[frec < umbral]
    return s.where(~s.isin(marginales), etiqueta)


def tabla_cardinalidad(antes: pd.DataFrame, despues: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Resumen por variable: categorías antes/después y % de filas que quedaron en 'OTHER'."""
    filas = []
    for c in cols:
        cambio = antes[c].to_numpy() != despues[c].to_numpy()
        filas.append({
            "variable": c,
            "categorias_originales": antes[c].nunique(dropna=False),
            "categorias_finales": despues[c].nunique(dropna=False),
            "categorias_agrupadas": antes[c].nunique(dropna=False) - despues[c].nunique(dropna=False) + 1
            if cambio.any() else 0,
            "pct_filas_en_OTHER": 100 * cambio.mean(),
        })
    return pd.DataFrame(filas).set_index("variable").round(2)


def codificar_dummies(df: pd.DataFrame, cols: list[str], referencia: str | None) -> tuple[pd.DataFrame, dict]:
    """Variables indicadoras 0/1. Por cada variable se descarta la indicadora de la categoría
    de referencia (`referencia` si existe; si no, la categoría más frecuente) para evitar la
    colinealidad perfecta (trampa de las variables dummy)."""
    bloques, refs = [], {}
    for c in cols:
        d = pd.get_dummies(df[c], prefix=c, prefix_sep="=", dtype=np.int8)
        if referencia is not None:
            ref = referencia if (df[c] == referencia).any() else df[c].mode().iloc[0]
            d = d.drop(columns=f"{c}={ref}")
            refs[c] = ref
        bloques.append(d)
    return pd.concat(bloques, axis=1), refs


def variables_continuas(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    m = cfg.matriz
    cont = df[m["continuas"]].astype(float).copy()
    for c in m["log1p"]:
        cont[c] = np.log1p(cont[c])
        cont = cont.rename(columns={c: f"log_{c}"})
    return cont


def construir_matriz(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Matriz n x p: continuas + binarias + indicadoras. Índice = GlobalEventID.
    Devuelve la matriz, las categóricas ya reducidas y las categorías de referencia."""
    m = cfg.matriz
    cats = df[m["categoricas"]].apply(reducir_cardinalidad, umbral=m["umbral_otros"], etiqueta=m["etiqueta_otros"])
    dummies, refs = codificar_dummies(cats, m["categoricas"], m["referencia_dummy"] or None)
    X = pd.concat([variables_continuas(df, cfg), df[m["binarias"]].astype(np.int8), dummies], axis=1)
    X.index = df["GlobalEventID"].to_numpy()
    X.index.name = "GlobalEventID"
    cats.index = X.index
    return X, cats, refs
