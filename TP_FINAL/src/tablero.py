"""Tablero (Dash): un menú con las 17 características y un boxplot de la elegida, por ritmo (FA, sin FA) y franja (día, noche).
Y una figura fija con las 17 a la vez (matplotlib), con las mismas cajas, y la prueba de día contra noche que muestran los dos.

Lee sólo el almacén, y cada boxplot muestra las dos comparaciones (FA contra sin FA, y día contra noche). Las cajas se
calculan acá (cuartiles y bigotes de Tukey) y al navegador sólo van cinco números por caja, no los valores de las
ventanas. Arriba de cada ritmo va el p de Wilcoxon de día contra noche (`wilcoxon_dia_noche`). El cuaderno lanza el
tablero con `app.run(jupyter_mode="inline", port=8050)`; al re-ejecutar esa celda, Dash apaga el servidor anterior del
mismo puerto. Este módulo no arranca ningún servidor ni elige puerto. La figura fija está porque el tablero sólo se ve con
el servidor corriendo: en GitHub o en el PDF del cuaderno esa celda sale vacía.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

import caracteristicas
import config

CARACTERISTICA_INICIAL = "P_RMSSD"
ORDEN = {"etiqueta": ["FA", "sin FA (N)"], "franja": ["día", "noche"]}      # FA antes que sin FA, y día antes que noche
ROTULOS = {"etiqueta": "Ritmo", "franja": "Franja"}
COLORES = {"día": "#eb6834", "noche": "#2a78d6"}                         # naranja el día y azul la noche, en los dos gráficos
NOTA_P = "p: Wilcoxon de día contra noche, pareado por paciente, con corrección de Holm"

COLUMNAS = ", ".join(f"f.{c}" for c in caracteristicas.CARACTERISTICAS)
CONSULTA = f"""SELECT h.holter_id, h.paciente_id, h.tipo_fa, r.etiqueta, t.franja, {COLUMNAS}
               FROM hecho_ventana f
               JOIN dim_holter h USING (holter_key)
               JOIN dim_tiempo t USING (tiempo_key)
               JOIN dim_ritmo  r USING (ritmo_key)"""


def leer_hechos(ruta: Path = config.ALMACEN) -> pd.DataFrame:
    """Una fila por ventana: `holter_id`, `paciente_id`, `tipo_fa`, `etiqueta` (ritmo), `franja` y las 17 características
    (NaN si son NULL).

    Une `hecho_ventana` con sus tres dimensiones con una sola consulta. Se hace una vez al armar el tablero: después sólo se
    redibuja el gráfico. El almacén se abre de sólo lectura (si falta, falla en vez de crearse vacío).
    """
    with closing(sqlite3.connect(f"{Path(ruta).resolve().as_uri()}?mode=ro", uri=True)) as con:
        return pd.read_sql_query(CONSULTA, con)


def cinco_numeros(valores: pd.Series) -> dict[str, float] | None:
    """Los cinco números de una caja, sin contar los NaN: Q1, mediana y Q3 (los de pandas, con interpolación lineal) y los
    bigotes de Tukey. Cada bigote llega al valor más lejano que esté a no más de 1,5 veces el RIC (Q3 - Q1) de la caja; los
    valores que quedan más lejos no se dibujan. Devuelve None si no hay ningún valor."""
    v = valores.dropna()
    if v.empty:
        return None
    q1, mediana, q3 = v.quantile([0.25, 0.5, 0.75]).tolist()
    ric = q3 - q1
    return {"q1": q1, "median": mediana, "q3": q3,
            "lowerfence": float(v[v >= q1 - 1.5 * ric].min()), "upperfence": float(v[v <= q3 + 1.5 * ric].max())}


def cajas_por_grupo(hechos: pd.DataFrame, caracteristica: str) -> dict[tuple[str, str], dict[str, float]]:
    """Los cinco números de `caracteristica` en cada grupo, {(ritmo, franja): números}, en el orden de ORDEN. Un grupo sin
    ningún valor (todo NaN, o sin ventanas) no está. Lo usan los dos gráficos, así las cajas son las mismas en ambos."""
    grupos = {}
    for etiqueta in ORDEN["etiqueta"]:
        for franja in ORDEN["franja"]:
            numeros = cinco_numeros(hechos.loc[(hechos["etiqueta"] == etiqueta) & (hechos["franja"] == franja),
                                               caracteristica])
            if numeros is not None:
                grupos[(etiqueta, franja)] = numeros
    return grupos


# --- Día contra noche: Wilcoxon pareado por paciente ------------------------------------------------------------------


def holm(p: pd.Series) -> pd.Series:
    """Corrección de Holm para comparaciones múltiples. Con m pruebas, el p más chico se multiplica por m, el siguiente por
    m - 1, y así hasta el más grande, que queda igual; después ninguno puede quedar por debajo del anterior y ninguno pasa
    de 1. Los NaN no cuentan como prueba y siguen siendo NaN."""
    validos = p.dropna().sort_values()
    ajustados = (validos * (len(validos) - np.arange(len(validos)))).cummax().clip(upper=1)
    return ajustados.reindex(p.index)


def wilcoxon_dia_noche(hechos: pd.DataFrame) -> pd.DataFrame:
    """Día contra noche, en cada ritmo y en cada característica: prueba de Wilcoxon de rangos con signo, pareada por paciente.

    La unidad es el paciente, no la ventana: las ventanas de un mismo paciente no son independientes. En cada ritmo, cada
    paciente que tiene ventanas de ese ritmo de día y de noche aporta un par: la mediana de sus ventanas de día y la de sus
    ventanas de noche (si tiene dos Holter, entran juntos). Los p se corrigen por Holm sobre todas las pruebas (17 por 2
    ritmos). Devuelve una fila por (`etiqueta`, `caracteristica`) con `n` (pacientes con par), `p` y `p_holm`. Si no hay
    pares, o todas las diferencias son cero, no hay prueba: p es NaN.
    """
    columnas = list(caracteristicas.CARACTERISTICAS)
    filas = []
    for etiqueta in ORDEN["etiqueta"]:
        del_ritmo = hechos[hechos["etiqueta"] == etiqueta]
        medianas = del_ritmo.groupby(["paciente_id", "franja"])[columnas].median()
        for caracteristica in columnas:
            pares = medianas[caracteristica].unstack("franja").reindex(columns=ORDEN["franja"]).dropna()
            dia, noche = pares["día"], pares["noche"]
            p = stats.wilcoxon(dia, noche).pvalue if (dia != noche).any() else float("nan")
            filas.append({"etiqueta": etiqueta, "caracteristica": caracteristica, "n": len(pares), "p": float(p)})
    pruebas = pd.DataFrame(filas).set_index(["etiqueta", "caracteristica"])
    pruebas["p_holm"] = holm(pruebas["p"])
    return pruebas


def texto_p(p: float) -> str:
    """El p como se muestra arriba de cada ritmo, con coma decimal: "p < 0,001", "p = 0,004", "p = 0,12" o "p: sin prueba"."""
    if pd.isna(p):
        return "p: sin prueba"
    if p < 0.001:
        return "p < 0,001"
    return f"p = {p:.3f}".replace(".", ",") if p < 0.01 else f"p = {p:.2f}".replace(".", ",")


# --- Gráficos ---------------------------------------------------------------------------------------------------------


def figura(hechos: pd.DataFrame, caracteristica: str, pruebas: pd.DataFrame | None = None) -> go.Figure:
    """Boxplot de `caracteristica` para el tablero: ritmo en el eje x y una caja de día y otra de noche en cada uno.

    Los cinco números de cada caja vienen de `cajas_por_grupo`: la figura no lleva los valores de las ventanas. Un grupo
    sin ningún valor queda sin caja. Siempre hay una traza para el día y otra para la noche, aunque queden vacías, así los
    colores no cambian de una característica a otra. Con `pruebas` (lo que devuelve `wilcoxon_dia_noche`), arriba de cada
    ritmo va su p corregido.
    """
    grupos = cajas_por_grupo(hechos, caracteristica)
    fig = go.Figure()
    for franja in ORDEN["franja"]:
        de_la_franja = {etiqueta: numeros for (etiqueta, f), numeros in grupos.items() if f == franja}
        fig.add_trace(go.Box(name=franja, x=list(de_la_franja), marker_color=COLORES[franja],
                             **{clave: [numeros[clave] for numeros in de_la_franja.values()]
                                for clave in ("q1", "median", "q3", "lowerfence", "upperfence")}))
    fig.update_layout(boxmode="group", yaxis_title=caracteristica, legend_title=ROTULOS["franja"],
                      xaxis={"title": ROTULOS["etiqueta"], "categoryorder": "array", "categoryarray": ORDEN["etiqueta"]})
    if pruebas is not None:
        for etiqueta in ORDEN["etiqueta"]:
            fig.add_annotation(x=etiqueta, y=1, yref="paper", yanchor="bottom", showarrow=False,
                               text=texto_p(pruebas.loc[(etiqueta, caracteristica), "p_holm"]))
    return fig


def figura_17(hechos: pd.DataFrame, pruebas: pd.DataFrame | None = None) -> Figure:
    """Las 17 características en una figura fija: un boxplot por característica, con las mismas cajas que `figura`.

    En cada panel van FA y sin FA (N) en el eje x, con la caja del día y la de la noche juntas. Los atípicos no se dibujan,
    como en el tablero. Un grupo sin ningún valor queda sin caja. Con `pruebas`, arriba de cada ritmo va su p corregido,
    como en el tablero.
    """
    fig, ejes = plt.subplots(5, 4, figsize=(11, 13))
    for ax, caracteristica in zip(ejes.flat, caracteristicas.CARACTERISTICAS):
        for (etiqueta, franja), numeros in cajas_por_grupo(hechos, caracteristica).items():
            posicion = 3 * ORDEN["etiqueta"].index(etiqueta) + ORDEN["franja"].index(franja)  # 0 y 1 FA; 3 y 4 sin FA
            ax.bxp([{"q1": numeros["q1"], "med": numeros["median"], "q3": numeros["q3"],
                     "whislo": numeros["lowerfence"], "whishi": numeros["upperfence"]}],
                   positions=[posicion], widths=0.8, showfliers=False, manage_ticks=False, patch_artist=True,
                   boxprops={"facecolor": COLORES[franja]}, medianprops={"color": "black"})
        ax.set_xlim(-0.8, 4.8)
        ax.set_xticks([0.5, 3.5], ORDEN["etiqueta"])
        ax.set_title(caracteristica, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
        if pruebas is not None:
            abajo, arriba = ax.get_ylim()
            ax.set_ylim(abajo, arriba + 0.15 * (arriba - abajo))      # lugar arriba de las cajas para el p
            for x, etiqueta in zip([0.5, 3.5], ORDEN["etiqueta"]):
                ax.text(x, 0.97, texto_p(pruebas.loc[(etiqueta, caracteristica), "p_holm"]), fontsize=8,
                        ha="center", va="top", transform=ax.get_xaxis_transform())
    for ax in ejes.flat[len(caracteristicas.CARACTERISTICAS):]:
        ax.axis("off")
    fig.legend(handles=[Patch(facecolor=COLORES[franja], edgecolor="black", label=franja) for franja in ORDEN["franja"]],
               title=ROTULOS["franja"], loc="lower right", bbox_to_anchor=(0.9, 0.08))
    fig.suptitle("Las 17 características por ritmo y franja (ventanas de 30 s)"
                 + (f"\n{NOTA_P}" if pruebas is not None else ""))
    fig.tight_layout()
    return fig


def crear_app(hechos: pd.DataFrame, mensaje: str, pruebas: pd.DataFrame | None = None) -> Dash:
    """El tablero: título, `mensaje`, el menú (`id="caracteristica"`) y el gráfico (`id="boxplot"`) que se redibuja al elegir.
    Con `pruebas`, el gráfico lleva el p de cada ritmo y abajo una nota que dice qué prueba es.

    Devuelve la app sin arrancarla: el cuaderno elige cómo y en qué puerto correrla.
    """
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Variabilidad del ritmo en SHDB-AF: FA y sin FA, de día y de noche"),
        html.P(mensaje),
        dcc.Dropdown(id="caracteristica", options=[{"label": c, "value": c} for c in caracteristicas.CARACTERISTICAS],
                     value=CARACTERISTICA_INICIAL, clearable=False),
        dcc.Graph(id="boxplot"),
        *([html.P(NOTA_P + ".")] if pruebas is not None else []),
    ])

    @app.callback(Output("boxplot", "figure"), Input("caracteristica", "value"))
    def redibujar(caracteristica: str) -> go.Figure:
        return figura(hechos, caracteristica, pruebas)

    return app
