"""Ingesta de datos crudos de GDELT 2.0 hacia data/raw.

- Los export.CSV.zip de un día se guardan tal cual en data/raw/gdelt/AAAAMMDD/ (caché:
  si ya están, no se vuelven a descargar).
- Las tablas de códigos (CAMEO, tipos de actor, países FIPS) se guardan en data/raw/lookups/.
- `leer_eventos` parsea los ZIP crudos con las columnas del codebook.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

# Las 61 columnas del export.CSV de GDELT 2.0 (no trae encabezado), según el Event Codebook V2.0.
_ACTOR = ["Code", "Name", "CountryCode", "KnownGroupCode", "EthnicCode",
          "Religion1Code", "Religion2Code", "Type1Code", "Type2Code", "Type3Code"]
_GEO = ["Type", "FullName", "CountryCode", "ADM1Code", "ADM2Code", "Lat", "Long", "FeatureID"]
COLUMNAS_EXPORT = (
    ["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate"]
    + [f"Actor1{c}" for c in _ACTOR]
    + [f"Actor2{c}" for c in _ACTOR]
    + ["IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
       "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]
    + [f"Actor1Geo_{c}" for c in _GEO]
    + [f"Actor2Geo_{c}" for c in _GEO]
    + [f"ActionGeo_{c}" for c in _GEO]
    + ["DATEADDED", "SOURCEURL"]
)
assert len(COLUMNAS_EXPORT) == 61

# Subconjunto que alimenta el almacén de datos
COLUMNAS_USADAS = [
    "GlobalEventID", "Day", "DATEADDED", "SOURCEURL", "IsRootEvent",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1Type1Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2Type1Code",
    "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_Lat", "ActionGeo_Long",
]
_NUMERICAS = {
    "GlobalEventID": "int64", "Day": "int64", "DATEADDED": "int64", "IsRootEvent": "int8",
    "QuadClass": "int8", "GoldsteinScale": "float64", "NumMentions": "int64",
    "NumSources": "int64", "NumArticles": "int64", "AvgTone": "float64",
    "ActionGeo_Type": "Int8", "ActionGeo_Lat": "float64", "ActionGeo_Long": "float64",
}

# nombre de archivo -> ¿tiene fila de encabezado?
LOOKUPS = {
    "CAMEO.eventcodes.txt": True,
    "CAMEO.type.txt": True,
    "FIPS.country.txt": False,
}


def crear_session(max_workers: int = 16) -> requests.Session:
    """Session HTTP con reintentos y backoff exponencial (igual criterio que en el TP1)."""
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=max_workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def urls_del_dia(cfg: Config) -> list[str]:
    """Las URLs de GDELT siguen el patrón AAAAMMDDHHMMSS.export.CSV.zip cada 15 minutos,
    así que se generan directamente en lugar de recorrer la masterfilelist (~130 MB)."""
    inicio = datetime.combine(cfg.fecha, datetime.min.time())
    n = cfg.ingesta["archivos_por_dia"]
    return [
        f"{cfg.ingesta['url_base']}/{(inicio + timedelta(minutes=15 * i)):%Y%m%d%H%M%S}.export.CSV.zip"
        for i in range(n)
    ]


def _descargar(session: requests.Session, url: str, destino: Path) -> str:
    if destino.exists() and destino.stat().st_size > 0:
        return "cache"
    r = session.get(url, timeout=120)
    if r.status_code == 404:
        return "faltante"
    r.raise_for_status()
    tmp = destino.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(destino)
    return "descargado"


def descargar_dia(cfg: Config) -> pd.DataFrame:
    """Descarga los export.CSV.zip del día a data/raw/gdelt/AAAAMMDD. Devuelve un reporte."""
    cfg.dir_gdelt.mkdir(parents=True, exist_ok=True)
    session = crear_session(cfg.ingesta["max_workers"])
    urls = urls_del_dia(cfg)
    destinos = [cfg.dir_gdelt / u.rsplit("/", 1)[-1] for u in urls]

    def tarea(par):
        url, dest = par
        try:
            return _descargar(session, url, dest)
        except requests.RequestException as e:
            return f"error: {type(e).__name__}"

    with ThreadPoolExecutor(max_workers=cfg.ingesta["max_workers"]) as ex:
        estados = list(ex.map(tarea, zip(urls, destinos)))

    reporte = pd.DataFrame({"archivo": [d.name for d in destinos], "estado": estados})
    return reporte


def descargar_lookups(cfg: Config) -> None:
    cfg.dir_lookups.mkdir(parents=True, exist_ok=True)
    session = crear_session()
    for nombre in LOOKUPS:
        _descargar(session, f"{cfg.ingesta['url_lookups']}/{nombre}", cfg.dir_lookups / nombre)


def leer_lookup(cfg: Config, nombre: str) -> pd.DataFrame:
    """Lee una tabla de códigos como texto (los códigos CAMEO tienen ceros a la izquierda)."""
    return pd.read_csv(cfg.dir_lookups / nombre, sep="\t", header=None,
                       skiprows=1 if LOOKUPS[nombre] else 0, names=["codigo", "descripcion"],
                       dtype=str, keep_default_na=False, encoding="latin-1")


def _leer_zip(ruta: Path) -> pd.DataFrame:
    with zipfile.ZipFile(ruta) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                         sep="\t", header=None, names=COLUMNAS_EXPORT, usecols=COLUMNAS_USADAS,
                         dtype=str, keep_default_na=False, quoting=3)
    return df


def leer_eventos(cfg: Config) -> pd.DataFrame:
    """Concatena todos los ZIP crudos del día y tipa las columnas numéricas.

    Si un GlobalEventID aparece en más de un archivo, se conserva la última versión
    (la de DATEADDED más reciente, con los contadores actualizados)."""
    archivos = sorted(cfg.dir_gdelt.glob("*.export.CSV.zip"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos crudos en {cfg.dir_gdelt}. Corré descargar_dia().")
    with ThreadPoolExecutor(max_workers=8) as ex:
        df = pd.concat(list(ex.map(_leer_zip, archivos)), ignore_index=True)

    df = df.replace("", pd.NA)
    for col, tipo in _NUMERICAS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(tipo)
    df = (df.sort_values("DATEADDED")
            .drop_duplicates("GlobalEventID", keep="last")
            .reset_index(drop=True))
    return df
