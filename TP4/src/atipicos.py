"""Detección de eventos atípicos multivariantes a partir del ACP.

Con k componentes retenidas, cada evento se evalúa con dos estadísticos complementarios:

- T² de Hotelling (dentro del modelo): distancia de Mahalanobis en el subespacio de las k
  componentes, T² = Σ_{j≤k} t_j² / λ_j. Marca eventos con una combinación extrema de las
  variables que el modelo sí representa.
- SPE o Q (fuera del modelo): error cuadrático de reconstrucción, Q = ||z − ẑ||² = Σ_{j>k} t_j².
  Marca eventos que rompen la estructura de correlación (lo que las k componentes no explican).

Límites al nivel α: T² ~ χ²_k (n grande) y Q con la aproximación de Jackson–Mudholkar.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .acp import ResultadoACP


def t2_hotelling(res: ResultadoACP, k: int) -> pd.Series:
    S = res.scores.to_numpy()[:, :k]
    return pd.Series((S**2 / res.autovalores[:k]).sum(axis=1), index=res.scores.index, name="T2")


def spe(res: ResultadoACP, k: int) -> pd.Series:
    """Con el ACP completo, el residuo de reconstrucción es la suma de los scores² de las
    componentes descartadas (las componentes son ortonormales)."""
    S = res.scores.to_numpy()[:, k:]
    return pd.Series((S**2).sum(axis=1), index=res.scores.index, name="Q")


def limite_t2(k: int, alfa: float) -> float:
    return float(chi2.ppf(1 - alfa, df=k))


def limite_q(res: ResultadoACP, k: int, alfa: float) -> float:
    """Límite de Jackson–Mudholkar (1979) a partir de los autovalores residuales."""
    lam = res.autovalores[k:]
    t1, t2, t3 = lam.sum(), (lam**2).sum(), (lam**3).sum()
    h0 = 1 - 2 * t1 * t3 / (3 * t2**2)
    z = norm.ppf(1 - alfa)
    return float(t1 * (z * np.sqrt(2 * t2 * h0**2) / t1 + 1 + t2 * h0 * (h0 - 1) / t1**2) ** (1 / h0))


def contribuciones_t2(res: ResultadoACP, k: int, filas: np.ndarray) -> pd.DataFrame:
    """Contribución de cada variable al T² de las observaciones `filas` (posiciones):
    c_v = z_v · Σ_j p_vj t_j / λ_j. Las contribuciones de una fila suman su T²."""
    P = res.pca.components_[:k].T                         # p x k
    T = res.scores.to_numpy()[filas, :k]                  # m x k
    C = res.Z[filas] * ((T / res.autovalores[:k]) @ P.T)  # m x p
    return pd.DataFrame(C, index=res.scores.index[filas], columns=res.columnas)


def contribuciones_q(res: ResultadoACP, k: int, filas: np.ndarray) -> pd.DataFrame:
    """Contribución de cada variable al SPE: residuo de reconstrucción al cuadrado."""
    P = res.pca.components_[:k].T
    Zf = res.Z[filas]
    E = Zf - (Zf @ P) @ P.T
    return pd.DataFrame(E**2, index=res.scores.index[filas], columns=res.columnas)


def _principales(C: pd.DataFrame, n: int = 3) -> pd.Series:
    """Las n variables con mayor contribución positiva de cada fila, como texto."""
    valores = C.to_numpy()
    orden = np.argsort(-valores, axis=1)[:, :n]
    cols = np.array(C.columns)
    total = valores.sum(axis=1, keepdims=True)
    pct = np.take_along_axis(valores, orden, axis=1) / total * 100
    return pd.Series([", ".join(f"{cols[o]} ({p:.0f}%)" for o, p in zip(fila, pf))
                      for fila, pf in zip(orden, pct)], index=C.index)


def detectar(res: ResultadoACP, k: int, alfa: float) -> pd.DataFrame:
    """T², Q, sus límites y la clasificación de cada evento."""
    lim_t2, lim_q = limite_t2(k, alfa), limite_q(res, k, alfa)
    d = pd.concat([t2_hotelling(res, k), spe(res, k)], axis=1)
    d["T2_rel"] = d["T2"] / lim_t2
    d["Q_rel"] = d["Q"] / lim_q
    d["atipico_T2"] = d["T2_rel"] > 1
    d["atipico_Q"] = d["Q_rel"] > 1
    d["tipo_atipico"] = np.select(
        [d["atipico_T2"] & d["atipico_Q"], d["atipico_T2"], d["atipico_Q"]],
        ["T2 y Q", "solo T2", "solo Q"], default="normal")
    # Puntaje para ordenar: cuántas veces supera el límite más exigido
    d["puntaje"] = d[["T2_rel", "Q_rel"]].max(axis=1)
    d.attrs.update(limite_T2=lim_t2, limite_Q=lim_q, k=k, alfa=alfa)
    return d


def explicar(res: ResultadoACP, k: int, deteccion: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Variables que más contribuyen a T² y a Q en los eventos atípicos."""
    pos = np.flatnonzero((deteccion["tipo_atipico"] != "normal").to_numpy())
    return pd.DataFrame({
        "variables_T2": _principales(contribuciones_t2(res, k, pos), n),
        "variables_Q": _principales(contribuciones_q(res, k, pos), n),
    })


def grafico_control(deteccion: pd.DataFrame, color: pd.Series | None = None, n_muestra: int = 40_000,
                    semilla: int = 42, ax=None):
    """Gráfico T² vs Q (escala log) con los límites al nivel α: los cuadrantes separan
    normales, atípicos dentro del modelo (T²), fuera del modelo (Q) y ambos."""
    ax = ax or plt.subplots(figsize=(11, 8))[1]
    rng = np.random.default_rng(semilla)
    d = deteccion
    # Se dibujan todos los atípicos y una muestra de los normales
    normales = np.flatnonzero((d["tipo_atipico"] == "normal").to_numpy())
    atip = np.flatnonzero((d["tipo_atipico"] != "normal").to_numpy())
    idx = np.concatenate([rng.choice(normales, size=min(n_muestra, len(normales)), replace=False), atip])
    sub = d.iloc[idx]
    grupos = (color.iloc[idx] if color is not None else sub["tipo_atipico"])
    for g in grupos.value_counts().index:
        m = (grupos == g).to_numpy()
        ax.scatter(sub["T2"].to_numpy()[m], sub["Q"].to_numpy()[m], s=4, alpha=0.35, label=str(g))
    ax.axvline(d.attrs["limite_T2"], color="k", ls="--", lw=1)
    ax.axhline(d.attrs["limite_Q"], color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"T² de Hotelling (k = {d.attrs['k']}) — límite {d.attrs['limite_T2']:.1f}")
    ax.set_ylabel(f"SPE / Q — límite {d.attrs['limite_Q']:.1f}")
    ax.set_title(f"Gráfico de control T² vs Q (α = {d.attrs['alfa']})")
    ax.legend(markerscale=4, fontsize=8)
    return ax


MOTIVOS = {
    "log_LagDays": "Desfase entre evento y publicación",
    "log_NumSources": "Cobertura inusual (muchas fuentes)",
    "log_NumMentions": "Cobertura inusual (muchas menciones)",
    "AvgTone": "Tono extremo",
    "GoldsteinScale": "Impacto Goldstein atípico",
    "IsRootEvent": "Combinación poco frecuente de categorías",
}


def motivo(explicacion: pd.DataFrame, deteccion: pd.DataFrame) -> pd.Series:
    """Motivo principal del atípico: la variable que más contribuye al estadístico que más
    supera su límite (T² o Q). Las indicadoras se agrupan como 'combinación poco frecuente'."""
    d = deteccion.loc[explicacion.index]
    col = np.where(d["T2_rel"] >= d["Q_rel"], explicacion["variables_T2"], explicacion["variables_Q"])
    principal = pd.Series(col, index=explicacion.index).str.split(" \\(").str[0]
    return principal.map(MOTIVOS).fillna("Combinación poco frecuente de categorías").rename("motivo")


COLUMNAS_EXPORT = [
    "DateAdded", "EventDate", "LagDays", "EventCode", "EventDesc", "EventRootDesc", "QuadClassDesc",
    "Actor1Code", "Actor1Name", "Actor1Type", "Actor2Code", "Actor2Name", "Actor2Type",
    "ActionGeo_FullName", "ActionGeo_CountryName", "GoldsteinScale", "AvgTone",
    "NumMentions", "NumSources", "NumArticles", "SourceDomain", "SOURCEURL",
]


def tabla_atipicos(eventos: pd.DataFrame, deteccion: pd.DataFrame, explicacion: pd.DataFrame) -> pd.DataFrame:
    """Tabla aparte con los eventos atípicos: atributos de la OBT limpia + estadísticos,
    variables que los explican y motivo. Ordenada por puntaje (más atípico primero)."""
    ev = eventos.set_index("GlobalEventID").loc[explicacion.index, COLUMNAS_EXPORT]
    est = deteccion.loc[explicacion.index, ["T2", "Q", "T2_rel", "Q_rel", "tipo_atipico", "puntaje"]]
    t = pd.concat([ev, est, explicacion, motivo(explicacion, deteccion)], axis=1)
    return t.sort_values("puntaje", ascending=False).rename_axis("GlobalEventID").reset_index()
