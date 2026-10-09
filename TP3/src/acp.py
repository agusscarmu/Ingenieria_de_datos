"""Análisis por Componentes Principales sobre la matriz de datos estandarizada."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ResultadoACP:
    pca: PCA
    scaler: StandardScaler
    scores: pd.DataFrame          # proyecciones de las observaciones (n x p)
    varianza: pd.DataFrame        # autovalor, % explicado y % acumulado por componente
    columnas: list[str]

    def cargas(self, k: int | None = None) -> pd.DataFrame:
        """Cargas = autovector * sqrt(autovalor). Con datos estandarizados son la correlación
        entre cada variable y la componente, por lo que son comparables entre variables."""
        k = k or len(self.columnas)
        L = self.pca.components_[:k].T * np.sqrt(self.pca.explained_variance_[:k])
        return pd.DataFrame(L, index=self.columnas, columns=[f"PC{i + 1}" for i in range(k)])

    def autovectores(self, k: int | None = None) -> pd.DataFrame:
        k = k or len(self.columnas)
        return pd.DataFrame(self.pca.components_[:k].T, index=self.columnas,
                            columns=[f"PC{i + 1}" for i in range(k)])


def ajustar_acp(X: pd.DataFrame) -> ResultadoACP:
    # Se descartan columnas constantes (desvío 0): no aportan varianza y no se pueden estandarizar
    X = X.loc[:, X.std() > 0]
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.values.astype(float))
    pca = PCA(svd_solver="full").fit(Z)
    nombres = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(pca.transform(Z), index=X.index, columns=nombres)
    varianza = pd.DataFrame({
        "autovalor": pca.explained_variance_,
        "pct_varianza": 100 * pca.explained_variance_ratio_,
        "pct_acumulado": 100 * np.cumsum(pca.explained_variance_ratio_),
    }, index=nombres).round(3)
    return ResultadoACP(pca, scaler, scores, varianza, X.columns.tolist())


def criterios_componentes(res: ResultadoACP, varianza_objetivo: float = 0.80) -> pd.Series:
    """Cantidad de componentes que sugiere cada criterio clásico."""
    v = res.varianza
    return pd.Series({
        "Kaiser (autovalor > 1)": int((v["autovalor"] > 1).sum()),
        f"Varianza acumulada >= {varianza_objetivo:.0%}": int((v["pct_acumulado"] < 100 * varianza_objetivo).sum() + 1),
        "Varianza media (Jolliffe, autovalor > 0.7)": int((v["autovalor"] > 0.7).sum()),
    }, name="n_componentes")


def scree_plot(res: ResultadoACP, varianza_objetivo: float = 0.80, ax=None):
    ax = ax or plt.subplots(figsize=(11, 5))[1]
    v = res.varianza
    x = np.arange(1, len(v) + 1)
    ax.bar(x, v["autovalor"], color="#4C72B0", alpha=0.8, label="Autovalor")
    ax.plot(x, v["autovalor"], "o-", color="#1f3b63", ms=4)
    ax.axhline(1, ls="--", color="gray", lw=1, label="Kaiser (autovalor = 1)")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Autovalor")
    ax.set_xticks(x)
    ax2 = ax.twinx()
    ax2.plot(x, v["pct_acumulado"], "s-", color="#C44E52", ms=4, label="% varianza acumulada")
    ax2.axhline(100 * varianza_objetivo, ls=":", color="#C44E52", lw=1)
    ax2.set_ylabel("% varianza acumulada")
    ax2.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("Gráfico de sedimentación (scree plot)")
    return ax


def heatmap_cargas(cargas: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(1.1 * cargas.shape[1] + 4, 0.35 * cargas.shape[0] + 2))[1]
    sns.heatmap(cargas, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3, cbar_kws={"label": "carga (correlación)"}, ax=ax)
    ax.set_title("Cargas de las componentes principales")
    return ax


def cargas_principales(cargas: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Para cada componente, las k variables con mayor |carga| (con su signo)."""
    filas = []
    for pc in cargas.columns:
        top = cargas[pc].reindex(cargas[pc].abs().sort_values(ascending=False).index).head(k)
        for rank, (var, val) in enumerate(top.items(), 1):
            filas.append({"componente": pc, "rank": rank, "variable": var, "carga": round(val, 3)})
    return pd.DataFrame(filas).pivot(index="rank", columns="componente", values=["variable", "carga"]) \
        .swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def outliers_t2(res: ResultadoACP, k: int, alfa: float = 0.001) -> pd.Series:
    """T² de Hotelling en el subespacio de las k primeras componentes:
    T² = sum(score_j² / autovalor_j). Bajo normalidad ~ chi²(k); se marcan los que superan
    el cuantil 1 - alfa. Devuelve el T² de cada observación (atributo 'umbral')."""
    S = res.scores.iloc[:, :k].values
    t2 = pd.Series((S**2 / res.pca.explained_variance_[:k]).sum(axis=1), index=res.scores.index, name="T2")
    t2.attrs["umbral"] = chi2.ppf(1 - alfa, df=k)
    return t2


def biplot(res: ResultadoACP, color: pd.Series | None = None, n_muestra: int = 6000,
           n_flechas: int = 10, outliers: pd.Series | None = None, semilla: int = 42, ax=None):
    """Observaciones en PC1 vs PC2 (muestra aleatoria) + flechas de las variables con mayor
    carga en el plano. Las flechas se reescalan al rango de los scores para que se vean."""
    ax = ax or plt.subplots(figsize=(12, 9))[1]
    rng = np.random.default_rng(semilla)
    n = len(res.scores)
    idx = rng.choice(n, size=min(n_muestra, n), replace=False)
    S = res.scores.iloc[idx, :2]

    if color is not None:
        c = color.iloc[idx]
        for nivel in c.value_counts().index:
            m = (c == nivel).to_numpy()
            ax.scatter(S.iloc[m, 0], S.iloc[m, 1], s=6, alpha=0.35, label=str(nivel))
    else:
        ax.scatter(S.iloc[:, 0], S.iloc[:, 1], s=6, alpha=0.3)

    if outliers is not None:
        mo = (outliers.iloc[idx] > outliers.attrs["umbral"]).to_numpy()
        ax.scatter(S.iloc[mo, 0], S.iloc[mo, 1], s=18, facecolors="none", edgecolors="k",
                   lw=0.6, label="outlier T²")

    L = res.cargas(2)
    top = (L**2).sum(axis=1).sort_values(ascending=False).head(n_flechas).index
    escala = 0.9 * np.abs(S.values).max(axis=0).min() / np.abs(L.loc[top].values).max()
    for var in top:
        x, y = L.loc[var] * escala
        ax.arrow(0, 0, x, y, color="#8B0000", width=0.01, head_width=0.12, length_includes_head=True)
        ax.text(x * 1.08, y * 1.08, var, color="#8B0000", fontsize=8, ha="center", va="center")

    v = res.varianza["pct_varianza"]
    ax.set_xlabel(f"PC1 ({v.iloc[0]:.1f}% de la varianza)")
    ax.set_ylabel(f"PC2 ({v.iloc[1]:.1f}% de la varianza)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("Biplot PC1 vs PC2")
    if color is not None or outliers is not None:
        ax.legend(title=color.name if color is not None else None, markerscale=3, fontsize=8, loc="best")
    return ax
