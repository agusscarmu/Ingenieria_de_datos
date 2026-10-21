"""One Big Table (OBT): los eventos de la ventana desnormalizados en una única tabla.

En lugar del esquema estrella del TP2/TP3, cada fila trae el evento con sus 61 columnas originales
y, al lado, las descripciones de todos los códigos (CAMEO, tipos de actor, países) y atributos
derivados de fecha y del medio. Así el análisis no necesita ningún JOIN.

La OBT conserva los nulos tal como vienen de GDELT: el tratamiento de faltantes se hace después
(src/limpieza.py) para que quede documentado y sea reversible.
"""
from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd

from .config import Config
from .ingesta import leer_lookup

QUADCLASS = {1: "Cooperación verbal", 2: "Cooperación material", 3: "Conflicto verbal", 4: "Conflicto material"}
EJE_COOP_CONFLICTO = {1: "Cooperación", 2: "Cooperación", 3: "Conflicto", 4: "Conflicto"}
EJE_VERBAL_MATERIAL = {1: "Verbal", 2: "Material", 3: "Verbal", 4: "Material"}
# Tipo de geocodificación según el codebook (0 = sin ubicación)
GEO_TYPE = {1: "COUNTRY", 2: "USSTATE", 3: "USCITY", 4: "WORLDCITY", 5: "WORLDSTATE"}


def _dominio(url) -> str | None:
    if not isinstance(url, str):
        return None
    return urlparse(url).netloc.lower().removeprefix("www.") or None


def construir_obt(cfg: Config, eventos: pd.DataFrame) -> pd.DataFrame:
    cameo = leer_lookup(cfg, "CAMEO.eventcodes.txt")
    tipos = leer_lookup(cfg, "CAMEO.type.txt")
    paises_cameo = leer_lookup(cfg, "CAMEO.country.txt")
    paises_fips = leer_lookup(cfg, "FIPS.country.txt")

    obt = eventos.copy()
    nuevas = {}

    # Fechas: Day = fecha del evento; DATEADDED = momento en que GDELT lo publicó (UTC)
    nuevas["EventDate"] = pd.to_datetime(obt["Day"].astype(str), format="%Y%m%d", errors="coerce")
    nuevas["DateAdded"] = pd.to_datetime(obt["DATEADDED"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    nuevas["LagDays"] = (nuevas["DateAdded"].dt.normalize() - nuevas["EventDate"]).dt.days.astype("Int64")
    nuevas["HourAdded"] = nuevas["DateAdded"].dt.hour.astype("Int8")
    nuevas["WeekdayAdded"] = nuevas["DateAdded"].dt.day_name()

    # Acción (CAMEO): código, base y raíz con su descripción
    nuevas["EventDesc"] = obt["EventCode"].map(cameo)
    nuevas["EventBaseDesc"] = obt["EventBaseCode"].map(cameo)
    nuevas["EventRootDesc"] = obt["EventRootCode"] + " " + obt["EventRootCode"].map(cameo).fillna("?")
    q = obt["QuadClass"].astype("Int64")
    nuevas["QuadClassDesc"] = q.map(QUADCLASS)
    nuevas["CoopConflictAxis"] = q.map(EJE_COOP_CONFLICTO)
    nuevas["VerbalMaterialAxis"] = q.map(EJE_VERBAL_MATERIAL)

    # Actores: tipo (CAMEO.type) y país (CAMEO.country, códigos de 3 letras)
    for n in (1, 2):
        nuevas[f"Actor{n}Type1Desc"] = obt[f"Actor{n}Type1Code"].map(tipos)
        nuevas[f"Actor{n}CountryName"] = obt[f"Actor{n}CountryCode"].map(paises_cameo)

    # Geografía: países FIPS 10-4 y tipo de geocodificación
    for g in ("Actor1Geo", "Actor2Geo", "ActionGeo"):
        nuevas[f"{g}_CountryName"] = obt[f"{g}_CountryCode"].map(paises_fips)
    nuevas["ActionGeo_TypeDesc"] = obt["ActionGeo_Type"].astype("Int64").map(GEO_TYPE)

    # Medio
    nuevas["SourceDomain"] = obt["SOURCEURL"].map(_dominio)

    return pd.concat([obt, pd.DataFrame(nuevas, index=obt.index)], axis=1)


def diccionario_obt(obt: pd.DataFrame) -> pd.DataFrame:
    """Columnas de la OBT: origen (codebook / derivada), tipo y un ejemplo."""
    from .ingesta import COLUMNAS_EXPORT
    return pd.DataFrame({
        "origen": ["export.CSV" if c in COLUMNAS_EXPORT else "derivada" for c in obt.columns],
        "tipo": obt.dtypes.astype(str).values,
        "ejemplo": [obt[c].dropna().iloc[0] if obt[c].notna().any() else None for c in obt.columns],
    }, index=obt.columns)
