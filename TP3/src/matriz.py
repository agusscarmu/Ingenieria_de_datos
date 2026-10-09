"""Construcción de la matriz de datos: reducción de cardinalidad, variables indicadoras
y unión con las variables continuas."""
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
        filas.append({
            "variable": c,
            "categorias_originales": antes[c].nunique(dropna=False),
            "categorias_finales": despues[c].nunique(dropna=False),
            "pct_filas_en_OTHER": 100 * (despues[c] != antes[c]).mean(),
        })
    return pd.DataFrame(filas).set_index("variable").round(2)


def codificar_dummies(df: pd.DataFrame, cols: list[str], referencia: str | None) -> tuple[pd.DataFrame, dict]:
    """Variables indicadoras 0/1. Por cada variable se descarta la indicadora de la categoría
    de referencia (`referencia` si existe; si no, la categoría más frecuente).
    Con referencia=None se conservan todas las indicadoras."""
    bloques, refs = [], {}
    for c in cols:
        d = pd.get_dummies(df[c], prefix=c, prefix_sep="=", dtype=np.int8)
        if referencia is not None:
            ref = referencia if (df[c] == referencia).any() else df[c].mode().iloc[0]
            d = d.drop(columns=f"{c}={ref}")
            refs[c] = ref
        bloques.append(d)
    return pd.concat(bloques, axis=1), refs


def variables_continuas(vista: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cont = vista[cfg.matriz["continuas"]].astype(float).copy()
    for c in cfg.matriz["log1p"]:
        cont[c] = np.log1p(cont[c])
        cont = cont.rename(columns={c: f"log_{c}"})
    return cont


def construir_matriz(vista: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Matriz n x p: continuas + es_evento_raiz + indicadoras. Índice = global_event_id."""
    m = cfg.matriz
    cats = vista[m["categoricas"]].fillna("SIN_DATO").apply(
        reducir_cardinalidad, umbral=m["umbral_otros"], etiqueta=m["etiqueta_otros"])
    dummies, _ = codificar_dummies(cats, m["categoricas"], m["referencia_dummy"] or None)
    X = pd.concat([
        variables_continuas(vista, cfg),
        vista[["es_evento_raiz"]].astype(np.int8),
        dummies,
    ], axis=1)
    X.index = vista["global_event_id"].to_numpy()
    X.index.name = "global_event_id"
    return X
