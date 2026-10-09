"""Construcción del almacén de datos (esquema estrella del TP2 extendido) en SQLite.

El almacén se regenera completo en cada corrida a partir de los datos crudos, por lo que
es una salida derivada (data/processed) y no la fuente de verdad.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .config import Config
from .ingesta import leer_lookup

ESQUEMA = Path(__file__).parent / "sql" / "dw_schema.sql"

TIPO_INTERACCION = pd.DataFrame({
    "interaccion_key": [1, 2, 3, 4],
    "descripcion": ["Cooperación verbal", "Cooperación material", "Conflicto verbal", "Conflicto material"],
    "eje_coop_conflicto": ["Cooperación", "Cooperación", "Conflicto", "Conflicto"],
    "eje_verbal_material": ["Verbal", "Material", "Verbal", "Material"],
})


def _dim_tiempo(fechas: pd.Series) -> pd.DataFrame:
    f = pd.to_datetime(pd.Series(fechas.dropna().unique()), format="%Y%m%d")
    return pd.DataFrame({
        "tiempo_key": f.dt.strftime("%Y%m%d").astype(int),
        "fecha": f.dt.strftime("%Y-%m-%d"),
        "anio": f.dt.year,
        "trimestre": f.dt.quarter,
        "mes": f.dt.month,
        "dia_mes": f.dt.day,
        "dia_semana": f.dt.dayofweek + 1,
        "es_fin_de_semana": (f.dt.dayofweek >= 5).astype(int),
    }).sort_values("tiempo_key")


def _dim_actor(ev: pd.DataFrame, tipos: dict[str, str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Una única dimensión para ambos roles. Devuelve la dimensión y las claves de actor 1 y 2."""
    nk = ["actor_code", "nombre", "pais_actor_code", "tipo_actor_code"]
    roles = {}
    for n in (1, 2):
        roles[n] = ev[[f"Actor{n}Code", f"Actor{n}Name", f"Actor{n}CountryCode", f"Actor{n}Type1Code"]]
        roles[n].columns = nk
    todos = pd.concat(roles.values()).dropna(subset=["actor_code"]).drop_duplicates()
    dim = todos.reset_index(drop=True)
    dim.insert(0, "actor_key", dim.index + 1)
    dim["tipo_actor_desc"] = dim["tipo_actor_code"].map(tipos)
    sin_actor = pd.DataFrame([{"actor_key": 0, "nombre": "Sin actor"}])
    dim = pd.concat([sin_actor, dim], ignore_index=True)

    claves = {}
    for n, r in roles.items():
        claves[n] = r.merge(dim[["actor_key", *nk]], on=nk, how="left")["actor_key"].fillna(0).astype(int)
        claves[n].index = ev.index
    return dim, claves[1], claves[2]


def _dim_ubicacion(ev: pd.DataFrame, paises: dict[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    nk = ["pais_code", "adm1_code", "nombre_completo", "tipo_geo"]
    g = ev[["ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_FullName", "ActionGeo_Type",
            "ActionGeo_Lat", "ActionGeo_Long"]].copy()
    g.columns = nk + ["latitud", "longitud"]
    g["tipo_geo"] = g["tipo_geo"].astype("Int64")
    tiene_geo = g["tipo_geo"].fillna(0) > 0
    dim = g[tiene_geo].drop_duplicates(subset=nk).reset_index(drop=True)
    dim.insert(0, "ubicacion_key", dim.index + 1)
    dim.insert(2, "pais_nombre", dim["pais_code"].map(paises))
    dim = pd.concat([pd.DataFrame([{"ubicacion_key": 0, "nombre_completo": "Sin ubicación"}]), dim],
                    ignore_index=True)
    claves = g[nk].merge(dim[["ubicacion_key", *nk]], on=nk, how="left")["ubicacion_key"]
    claves.index = ev.index
    return dim, claves.where(tiene_geo, 0).fillna(0).astype(int)


def _dim_cameo(ev: pd.DataFrame, cameo: dict[str, str]) -> pd.DataFrame:
    dim = (ev[["EventCode", "EventBaseCode", "EventRootCode"]].drop_duplicates("EventCode")
           .rename(columns={"EventCode": "cameo_code", "EventBaseCode": "codigo_base",
                            "EventRootCode": "codigo_raiz"})
           .sort_values("cameo_code").reset_index(drop=True))
    dim.insert(0, "cameo_key", dim.index + 1)
    dim["descripcion"] = dim["cameo_code"].map(cameo)
    dim["descripcion_base"] = dim["codigo_base"].map(cameo)
    dim["descripcion_raiz"] = dim["codigo_raiz"].map(cameo)
    return dim


def _dominio(url) -> str:
    if not isinstance(url, str):
        return "desconocido"
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.") or "desconocido"


def construir_almacen(cfg: Config, eventos: pd.DataFrame) -> dict[str, int]:
    """Puebla el esquema estrella a partir de los eventos crudos. Devuelve filas por tabla."""
    ev = eventos.dropna(subset=["GoldsteinScale", "AvgTone", "EventCode", "QuadClass"])
    ev = ev[ev["QuadClass"].between(1, 4)].copy()
    ev["fecha_pub"] = ev["DATEADDED"] // 1_000_000
    ev = ev[pd.to_datetime(ev["Day"].astype(str), format="%Y%m%d", errors="coerce").notna()]

    cameo = dict(leer_lookup(cfg, "CAMEO.eventcodes.txt").values)
    tipos = dict(leer_lookup(cfg, "CAMEO.type.txt").values)
    paises = dict(leer_lookup(cfg, "FIPS.country.txt").values)

    dim_tiempo = _dim_tiempo(pd.concat([ev["Day"], ev["fecha_pub"]]).astype(str))
    dim_actor, a1, a2 = _dim_actor(ev, tipos)
    dim_ubic, ubic = _dim_ubicacion(ev, paises)
    dim_cameo = _dim_cameo(ev, cameo)
    dominios = ev["SOURCEURL"].map(_dominio)
    dim_medio = pd.DataFrame({"dominio": sorted(dominios.unique())})
    dim_medio.insert(0, "medio_key", dim_medio.index + 1)

    fact = pd.DataFrame({
        "global_event_id": ev["GlobalEventID"],
        "fecha_publicacion_key": ev["fecha_pub"],
        "fecha_evento_key": ev["Day"],
        "actor1_key": a1,
        "actor2_key": a2,
        "ubicacion_key": ubic,
        "cameo_key": ev["EventCode"].map(dim_cameo.set_index("cameo_code")["cameo_key"]),
        "interaccion_key": ev["QuadClass"].astype(int),
        "medio_key": dominios.map(dim_medio.set_index("dominio")["medio_key"]),
        "url_noticia": ev["SOURCEURL"],
        "es_evento_raiz": ev["IsRootEvent"].astype(int),
        "num_mentions": ev["NumMentions"],
        "num_sources": ev["NumSources"],
        "num_articles": ev["NumArticles"],
        "avg_tone": ev["AvgTone"],
        "goldstein_scale": ev["GoldsteinScale"],
    })

    ruta = cfg.ruta("almacen")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.unlink(missing_ok=True)
    tablas = {
        "dim_tiempo": dim_tiempo, "dim_actor": dim_actor, "dim_cameo": dim_cameo,
        "dim_tipo_interaccion": TIPO_INTERACCION, "dim_ubicacion": dim_ubic,
        "dim_medio": dim_medio, "fact_cobertura_evento": fact,
    }
    with sqlite3.connect(ruta) as con:
        con.executescript(ESQUEMA.read_text())
        for nombre, df in tablas.items():
            df.to_sql(nombre, con, if_exists="append", index=False, chunksize=50_000)
        huerfanas = con.execute("PRAGMA foreign_key_check").fetchall()
    if huerfanas:
        raise RuntimeError(f"Violaciones de integridad referencial: {huerfanas[:5]}")
    return {n: len(df) for n, df in tablas.items()}


def leer_vista_analisis(cfg: Config) -> pd.DataFrame:
    with sqlite3.connect(cfg.ruta("almacen")) as con:
        return pd.read_sql("SELECT * FROM v_eventos_analisis", con)
