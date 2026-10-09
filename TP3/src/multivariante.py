"""Descripción multivariante: vectores de medias, distancias y matrices de correlación."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist


def paises_principales(grupo: pd.Series, top: int, minimo: int,
                       excluir: tuple[str, ...] = ("SIN_UBICACION",)) -> list[str]:
    frec = grupo[~grupo.isin(excluir)].value_counts()
    return frec[frec >= minimo].head(top).index.tolist()


def vectores_medias(X: pd.DataFrame, grupo: pd.Series, paises: list[str]) -> pd.DataFrame:
    """Filas: GLOBAL + un vector de medias por país. Columnas: variables de X."""
    por_pais = X[grupo.isin(paises).to_numpy()].groupby(grupo[grupo.isin(paises)].to_numpy()).mean()
    por_pais = por_pais.loc[paises]
    medias = pd.concat([X.mean().to_frame("GLOBAL").T, por_pais])
    medias.index.name = "perfil"
    return medias


def matriz_distancias(medias: pd.DataFrame, X: pd.DataFrame, metrica: str = "euclidea_std") -> pd.DataFrame:
    """Distancias entre todos los vectores de medias (incluye GLOBAL).

    - 'euclidea_std': euclídea sobre las medias estandarizadas con el desvío global de cada
      variable (cada variable aporta en unidades de su propia dispersión).
    - 'mahalanobis': usa la covarianza global de X (pseudo-inversa: las indicadoras pueden
      dejar la covarianza casi singular), descontando la redundancia entre variables correlacionadas.
    """
    if metrica == "euclidea_std":
        Z = medias / X.std(ddof=1)
        D = cdist(Z.values, Z.values, metric="euclidean")
    elif metrica == "mahalanobis":
        VI = np.linalg.pinv(np.cov(X.values, rowvar=False))
        D = cdist(medias.values, medias.values, metric="mahalanobis", VI=VI)
    else:
        raise ValueError(metrica)
    return pd.DataFrame(D, index=medias.index, columns=medias.index)


def pares_extremos(D: pd.DataFrame, k: int = 5, incluir_global: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Los k pares de perfiles más similares y más disímiles."""
    idx = [i for i in D.index if incluir_global or i != "GLOBAL"]
    sub = D.loc[idx, idx]
    tri = np.triu(np.ones(sub.shape, dtype=bool), k=1)
    pares = (sub.where(tri).rename_axis(index="perfil_a", columns="perfil_b")
             .stack().dropna().rename("distancia").reset_index()
             .sort_values("distancia"))
    return pares.head(k).reset_index(drop=True), pares.tail(k).iloc[::-1].reset_index(drop=True)


def contribucion_distancia(medias: pd.DataFrame, X: pd.DataFrame, a: str, b: str, k: int = 8) -> pd.Series:
    """Qué variables explican la distancia euclídea estandarizada entre dos perfiles
    (porcentaje de la distancia al cuadrado que aporta cada variable)."""
    z = (medias.loc[a] - medias.loc[b]) / X.std(ddof=1)
    aporte = 100 * z**2 / (z**2).sum()
    return aporte.sort_values(ascending=False).head(k).round(1)


def correlaciones(X: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Pearson. Entre continua e indicadora equivale a la punto-biserial y
    entre dos indicadoras al coeficiente phi. Las variables constantes quedan en NaN."""
    return X.corr()


def heatmap_correlacion(C: pd.DataFrame, titulo: str, ax=None, anotar: bool = False):
    ax = ax or plt.subplots(figsize=(11, 9))[1]
    mascara = np.triu(np.ones_like(C, dtype=bool), k=1)
    sns.heatmap(C, mask=mascara, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
                annot=anotar, fmt=".2f", annot_kws={"size": 6}, linewidths=0.3,
                cbar_kws={"shrink": 0.6, "label": "r de Pearson"}, ax=ax)
    ax.set_title(titulo)
    ax.tick_params(labelsize=7)
    return ax


def pares_correlacionados(C: pd.DataFrame, k: int = 10, umbral: float = 0.0) -> pd.DataFrame:
    """Pares de variables ordenados por |r| (sin la diagonal)."""
    tri = np.triu(np.ones(C.shape, dtype=bool), k=1)
    p = C.where(tri).rename_axis(index="var_a", columns="var_b").stack().dropna().rename("r").reset_index()
    p = p[p["r"].abs() >= umbral]
    return p.reindex(p["r"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


def comparar_correlaciones(X: pd.DataFrame, grupo: pd.Series, paises: list[str],
                           var_a: str, var_b: str) -> pd.Series:
    """Correlación entre dos variables dentro de cada país (y global) para comparar."""
    out = {"GLOBAL": X[var_a].corr(X[var_b])}
    for p in paises:
        sub = X[(grupo == p).to_numpy()]
        out[p] = sub[var_a].corr(sub[var_b])
    return pd.Series(out, name=f"r({var_a}, {var_b})").round(3)


def clustermap_distancias(D: pd.DataFrame, titulo: str, metodo: str = "average"):
    """Mapa de calor de distancias reordenado por un agrupamiento jerárquico (dendrograma)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    Z = linkage(squareform(D.values, checks=False), method=metodo)
    g = sns.clustermap(D, row_linkage=Z, col_linkage=Z, cmap="viridis_r", annot=True, fmt=".2f",
                       annot_kws={"size": 7}, figsize=(10, 9), cbar_kws={"label": "distancia"})
    g.figure.suptitle(titulo, y=1.02)
    return g


def heatmap_desvios_medias(medias: pd.DataFrame, X: pd.DataFrame, ax=None):
    """Desvío de cada vector de medias respecto del global, en desvíos estándar de cada variable."""
    Z = (medias - medias.loc["GLOBAL"]) / X.std(ddof=1)
    ax = ax or plt.subplots(figsize=(14, 6))[1]
    lim = np.nanmax(np.abs(Z.values))
    sns.heatmap(Z, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, annot=True, fmt=".2f",
                annot_kws={"size": 6}, linewidths=0.3, cbar_kws={"label": "(media país − media global) / DE"}, ax=ax)
    ax.set_title("Perfil multivariante: desvío estandarizado de cada país respecto del vector global")
    ax.tick_params(labelsize=7)
    return ax
