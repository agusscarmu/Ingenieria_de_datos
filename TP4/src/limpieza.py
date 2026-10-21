"""Limpieza de la OBT: diagnóstico de nulos y ceros, descarte e imputación.

Criterio general (ver notebook, sección 2.1):
- Nulo estructural: la ausencia tiene significado (p. ej. el evento no tiene actor 2). No se imputa:
  se recodifica como una categoría explícita (SIN_ACTOR, SIN_TIPO, SIN_UBICACION).
- Nulo real: el dato debería existir y falta. Se imputa si hay una regla del codebook que lo
  determine; si no, se descarta la fila cuando la variable es obligatoria para el análisis.
- Cero estructural: un 0 que en realidad codifica "sin dato" (ActionGeo_Type = 0) se trata como nulo.
- Cero válido: un 0 que es un valor de la escala (Goldstein 0, tono neutro, latitud 0 en el ecuador).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .ingesta import COLUMNAS_EXPORT, leer_lookup


def perfil_faltantes(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Por columna: % de nulos y, si es numérica, % de ceros."""
    columnas = columnas or list(df.columns)
    filas = []
    for c in columnas:
        s = df[c]
        numerica = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        filas.append({
            "columna": c,
            "tipo": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "pct_nulos": 100 * s.isna().mean(),
            "ceros": int((s == 0).sum()) if numerica else np.nan,
            "pct_ceros": 100 * (s == 0).mean() if numerica else np.nan,
        })
    return pd.DataFrame(filas).set_index("columna").round(3)


def columnas_a_descartar(perfil: pd.DataFrame, umbral: float) -> list[str]:
    """Columnas del export.CSV con una proporción de nulos mayor a `umbral`."""
    p = perfil.loc[perfil.index.isin(COLUMNAS_EXPORT)]
    return p.index[p["pct_nulos"] > 100 * umbral].tolist()


def imputar_goldstein(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """GoldsteinScale es una función del código CAMEO (tabla CAMEO.goldsteinscale.txt).
    Si falta, se toma el valor del código; si el código no está en la tabla, el del código base
    y luego el del código raíz. Devuelve la serie imputada y la máscara de valores imputados."""
    tabla = {k: float(v) for k, v in leer_lookup(cfg, "CAMEO.goldsteinscale.txt").items()}
    faltan = df["GoldsteinScale"].isna()
    valor = df["EventCode"].map(tabla)
    valor = valor.fillna(df["EventBaseCode"].map(tabla)).fillna(df["EventRootCode"].map(tabla))
    return df["GoldsteinScale"].fillna(valor.where(faltan)), faltan


def tipo_actor(df: pd.DataFrame, n: int) -> pd.Series:
    """Tipo del actor n con los dos nulos estructurales diferenciados:
    SIN_ACTOR (GDELT no identificó actor) vs SIN_TIPO (hay actor, pero sin Type1Code:
    típicamente un país o una entidad genérica)."""
    t = df[f"Actor{n}Type1Desc"].fillna(df[f"Actor{n}Type1Code"])
    t = t.where(df[f"Actor{n}Code"].notna(), "SIN_ACTOR")
    return t.fillna("SIN_TIPO").astype(str)


def ubicacion_accion(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """País y tipo de geocodificación de la acción. ActionGeo_Type = 0 es un cero estructural
    (sin ubicación), igual que ActionGeo_CountryCode nulo: ambos pasan a SIN_UBICACION."""
    sin_geo = df["ActionGeo_Type"].fillna(0).eq(0) | df["ActionGeo_CountryCode"].isna()
    pais = df["ActionGeo_CountryName"].fillna(df["ActionGeo_CountryCode"]).where(~sin_geo, "SIN_UBICACION")
    tipo = df["ActionGeo_TypeDesc"].where(~sin_geo, "SIN_UBICACION")
    return pais.astype(str), tipo.astype(str)


def near_zero_variance(df: pd.DataFrame, columnas: list[str], cv_minimo: float) -> pd.DataFrame:
    """Coeficiente de variación (σ/μ) y razón entre la moda y el segundo valor más frecuente."""
    filas = []
    for c in columnas:
        s = df[c].astype(float)
        vc = s.value_counts()
        filas.append({
            "variable": c,
            "media": s.mean(),
            "desvio": s.std(),
            "cv": s.std() / abs(s.mean()) if s.mean() != 0 else np.inf,
            "pct_moda": 100 * vc.iloc[0] / len(s),
            "razon_frecuencias": vc.iloc[0] / vc.iloc[1] if len(vc) > 1 else np.inf,
            "valores_unicos": len(vc),
        })
    t = pd.DataFrame(filas).set_index("variable")
    t["nzv_por_cv"] = t["cv"] < cv_minimo
    return t.round(3)


def limpiar(obt: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Aplica los criterios de limpieza. Devuelve la tabla limpia y un registro de lo realizado."""
    lc = cfg.limpieza
    registro = {"filas_iniciales": len(obt)}
    df = obt.copy()

    # 1) Columnas casi vacías (siguen en la OBT, no entran al análisis)
    perfil = perfil_faltantes(df, COLUMNAS_EXPORT)
    descartadas = columnas_a_descartar(perfil, lc["umbral_nulos_columna"])
    df = df.drop(columns=descartadas)
    registro["columnas_descartadas"] = descartadas

    # 2) Imputación de Goldstein a partir de la tabla del codebook
    df["GoldsteinScale"], imputados = imputar_goldstein(df, cfg)
    registro["goldstein_imputados"] = int(imputados.sum())
    df["GoldsteinImputed"] = imputados.astype(np.int8)

    # 3) Fechas inconsistentes: evento posterior a su publicación o fecha inválida
    lag_invalido = df["LagDays"].isna() | (df["LagDays"] < 0)
    registro["filas_fecha_invalida"] = int(lag_invalido.sum())
    df = df[~lag_invalido]

    # 4) Obligatorias: si alguna sigue nula después de imputar, la fila se descarta
    oblig = [c for c in lc["obligatorias"] if c in df.columns]
    faltan = df[oblig].isna().any(axis=1)
    registro["filas_sin_obligatorias"] = int(faltan.sum())
    df = df[~faltan]

    # 5) Nulos estructurales en categóricas -> categorías explícitas
    df["Actor1Type"] = tipo_actor(df, 1)
    df["Actor2Type"] = tipo_actor(df, 2)
    df["ActionGeo_CountryName"], df["ActionGeo_TypeDesc"] = ubicacion_accion(df)

    registro["filas_finales"] = len(df)
    return df.reset_index(drop=True), registro
