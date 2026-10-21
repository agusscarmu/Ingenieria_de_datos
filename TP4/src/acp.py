"""Análisis por Componentes Principales sobre la matriz de datos estandarizada."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ResultadoACP:
    pca: PCA
    scaler: StandardScaler
    Z: np.ndarray                 # matriz estandarizada (n x p)
    scores: pd.DataFrame          # proyecciones de las observaciones (n x p)
    varianza: pd.DataFrame        # autovalor, % explicado y % acumulado por componente
    columnas: list[str]

    @property
    def autovalores(self) -> np.ndarray:
        return self.pca.explained_variance_

    def cargas(self, k: int | None = None) -> pd.DataFrame:
        """Cargas = autovector * sqrt(autovalor). Con datos estandarizados son la correlación
        entre cada variable y la componente, por lo que son comparables entre variables."""
        k = k or len(self.columnas)
        L = self.pca.components_[:k].T * np.sqrt(self.autovalores[:k])
        return pd.DataFrame(L, index=self.columnas, columns=[f"PC{i + 1}" for i in range(k)])


def ajustar_acp(X: pd.DataFrame) -> ResultadoACP:
    # Las columnas constantes (desvío 0) no aportan varianza y no se pueden estandarizar
    X = X.loc[:, X.std() > 0]
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.to_numpy(dtype=float))
    pca = PCA(svd_solver="full").fit(Z)
    nombres = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(pca.transform(Z), index=X.index, columns=nombres)
    varianza = pd.DataFrame({
        "autovalor": pca.explained_variance_,
        "pct_varianza": 100 * pca.explained_variance_ratio_,
        "pct_acumulado": 100 * np.cumsum(pca.explained_variance_ratio_),
    }, index=nombres)
    return ResultadoACP(pca, scaler, Z, scores, varianza, X.columns.tolist())


def analisis_paralelo(Z: np.ndarray, n_muestra: int = 50_000, repeticiones: int = 20,
                      percentil: float = 95, semilla: int = 42) -> np.ndarray:
    """Análisis paralelo de Horn: autovalores de matrices de correlación de datos sin estructura.
    Cada columna se permuta de forma independiente, lo que conserva su distribución marginal
    (clave con indicadoras 0/1) pero destruye las correlaciones. Devuelve el percentil de los
    autovalores obtenidos en las `repeticiones`."""
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(Z), size=min(n_muestra, len(Z)), replace=False)
    base = Z[idx]
    autov = []
    for _ in range(repeticiones):
        perm = np.column_stack([rng.permutation(base[:, j]) for j in range(base.shape[1])])
        autov.append(np.linalg.eigvalsh(np.corrcoef(perm, rowvar=False))[::-1])
    return np.percentile(np.array(autov), percentil, axis=0)


def criterios_componentes(res: ResultadoACP, varianza_objetivo: float = 0.80,
                          horn: np.ndarray | None = None) -> pd.Series:
    """Cantidad de componentes que sugiere cada criterio."""
    v = res.varianza
    criterios = {
        "Kaiser (autovalor > 1)": int((v["autovalor"] > 1).sum()),
        f"Varianza acumulada >= {varianza_objetivo:.0%}": int((v["pct_acumulado"] < 100 * varianza_objetivo).sum() + 1),
        "Jolliffe (autovalor > 0.7)": int((v["autovalor"] > 0.7).sum()),
    }
    if horn is not None:
        # primeras componentes consecutivas cuyo autovalor supera al de los datos permutados
        supera = v["autovalor"].to_numpy() > horn
        criterios["Análisis paralelo de Horn"] = int(np.argmin(supera)) if not supera.all() else len(supera)
    return pd.Series(criterios, name="n_componentes")


def scree_plot(res: ResultadoACP, varianza_objetivo: float = 0.80, horn: np.ndarray | None = None, ax=None):
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    v = res.varianza
    x = np.arange(1, len(v) + 1)
    ax.bar(x, v["autovalor"], color="#4C72B0", alpha=0.8, label="Autovalor")
    ax.plot(x, v["autovalor"], "o-", color="#1f3b63", ms=4)
    ax.axhline(1, ls="--", color="gray", lw=1, label="Kaiser (autovalor = 1)")
    if horn is not None:
        ax.plot(x, horn, "^--", color="#55A868", ms=4, label="Horn (P95 datos permutados)")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Autovalor")
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, v["pct_acumulado"], "s-", color="#C44E52", ms=4, label="% varianza acumulada")
    ax2.axhline(100 * varianza_objetivo, ls=":", color="#C44E52", lw=1)
    ax2.set_ylabel("% varianza acumulada")
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9)
    ax.set_title("Gráfico de sedimentación (scree plot)")
    return ax


def heatmap_cargas(cargas: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(1.0 * cargas.shape[1] + 5, 0.33 * cargas.shape[0] + 2))[1]
    sns.heatmap(cargas, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3, cbar_kws={"label": "carga (correlación)"}, ax=ax)
    ax.set_title("Cargas de las componentes principales retenidas")
    return ax


def cargas_principales(cargas: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Para cada componente, las k variables con mayor |carga| (con su signo)."""
    filas = []
    for pc in cargas.columns:
        top = cargas[pc].reindex(cargas[pc].abs().sort_values(ascending=False).index).head(k)
        filas.append({"componente": pc,
                      **{f"#{r}": f"{var} ({val:+.2f})" for r, (var, val) in enumerate(top.items(), 1)}})
    return pd.DataFrame(filas).set_index("componente")
