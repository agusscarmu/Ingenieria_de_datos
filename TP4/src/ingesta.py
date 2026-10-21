"""Ingesta de datos crudos de GDELT 2.0 hacia data/raw.

- El manifiesto de la ventana (URL, tamaño y MD5 de cada export.CSV.zip) se arma a partir de la
  masterfilelist.txt. Como el archivo pesa ~130 MB y está ordenado cronológicamente, se pide solo
  la cola con un HTTP Range y se agranda el pedido hasta cubrir el inicio de la ventana.
- Los export.CSV.zip se guardan tal cual en data/raw/gdelt/<inicio>_<fin>/ (caché) y se verifica
  su MD5 contra el manifiesto.
- Las tablas de códigos (CAMEO, tipos de actor, países) se guardan en data/raw/lookups/.
- `leer_eventos` parsea los ZIP con las 61 columnas del codebook usando pyarrow.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
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

# Tipos numéricos según el codebook; el resto queda como texto (los códigos CAMEO tienen ceros
# a la izquierda y no deben convertirse a número).
TIPOS_NUMERICOS = {
    "GlobalEventID": pa.int64(), "Day": pa.int64(), "MonthYear": pa.int64(), "Year": pa.int64(),
    "FractionDate": pa.float64(), "IsRootEvent": pa.int8(), "QuadClass": pa.int8(),
    "GoldsteinScale": pa.float64(), "NumMentions": pa.int64(), "NumSources": pa.int64(),
    "NumArticles": pa.int64(), "AvgTone": pa.float64(), "DATEADDED": pa.int64(),
    **{f"{g}_{c}": t for g in ("Actor1Geo", "Actor2Geo", "ActionGeo")
       for c, t in (("Type", pa.int8()), ("Lat", pa.float64()), ("Long", pa.float64()))},
}

# nombre de archivo -> ¿tiene fila de encabezado?
LOOKUPS = {
    "CAMEO.eventcodes.txt": True,
    "CAMEO.type.txt": True,
    "CAMEO.country.txt": True,
    "CAMEO.goldsteinscale.txt": True,
    "FIPS.country.txt": False,
}

_PATRON_EXPORT = re.compile(r"/(\d{14})\.export\.CSV\.zip$")


def crear_session(max_workers: int = 16) -> requests.Session:
    """Session HTTP con reintentos y backoff exponencial (mismo criterio que en el TP1)."""
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD"))
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=max_workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parsear_manifiesto(texto: str) -> pd.DataFrame:
    filas = []
    for linea in texto.splitlines():
        partes = linea.split(" ")
        if len(partes) != 3:
            continue
        m = _PATRON_EXPORT.search(partes[2])
        if m:
            filas.append((partes[2], int(partes[0]), partes[1], datetime.strptime(m.group(1), "%Y%m%d%H%M%S")))
    return pd.DataFrame(filas, columns=["url", "bytes", "md5", "timestamp"])


def construir_manifiesto(cfg: Config, bytes_cola: int = 2_000_000) -> pd.DataFrame:
    """Lista de export.CSV.zip de la ventana tomada de la masterfilelist (cola por HTTP Range).

    La primera línea de la cola puede venir cortada, por eso se descarta antes de parsear."""
    inicio, fin = cfg.ventana
    session = crear_session()
    url = cfg.ingesta["url_masterfilelist"]
    total = int(session.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    while True:
        desde = max(0, total - bytes_cola)
        r = session.get(url, headers={"Range": f"bytes={desde}-"}, timeout=120)
        r.raise_for_status()
        texto = r.text if desde == 0 else r.text.split("\n", 1)[1]
        man = _parsear_manifiesto(texto)
        if desde == 0 or man["timestamp"].min() < inicio:
            break
        bytes_cola *= 4
    man = man[(man["timestamp"] >= inicio) & (man["timestamp"] < fin)]
    man = man.drop_duplicates("url").sort_values("timestamp").reset_index(drop=True)
    cfg.ruta("manifiesto").parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.ruta("manifiesto"), index=False)
    return man


def _md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def _descargar(session: requests.Session, url: str, destino: Path, md5: str | None = None) -> str:
    if destino.exists() and destino.stat().st_size > 0 and (md5 is None or _md5(destino) == md5):
        return "cache"
    r = session.get(url, timeout=120)
    if r.status_code == 404:
        return "faltante"
    r.raise_for_status()
    if md5 is not None and hashlib.md5(r.content).hexdigest() != md5:
        return "md5_invalido"
    tmp = destino.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(destino)
    return "descargado"


def descargar_ventana(cfg: Config, manifiesto: pd.DataFrame) -> pd.DataFrame:
    """Descarga los export.CSV.zip del manifiesto verificando el MD5. Devuelve un reporte por archivo."""
    cfg.dir_gdelt.mkdir(parents=True, exist_ok=True)
    session = crear_session(cfg.ingesta["max_workers"])

    def tarea(fila) -> str:
        destino = cfg.dir_gdelt / fila.url.rsplit("/", 1)[-1]
        try:
            return _descargar(session, fila.url, destino, fila.md5)
        except requests.RequestException as e:
            return f"error: {type(e).__name__}"

    with ThreadPoolExecutor(max_workers=cfg.ingesta["max_workers"]) as ex:
        estados = list(ex.map(tarea, manifiesto.itertuples(index=False)))
    return manifiesto.assign(estado=estados)


def descargar_lookups(cfg: Config) -> None:
    cfg.dir_lookups.mkdir(parents=True, exist_ok=True)
    session = crear_session()
    for nombre in LOOKUPS:
        _descargar(session, f"{cfg.ingesta['url_lookups']}/{nombre}", cfg.dir_lookups / nombre)


def leer_lookup(cfg: Config, nombre: str) -> dict[str, str]:
    """Tabla de códigos como diccionario código -> descripción (todo como texto)."""
    df = pd.read_csv(cfg.dir_lookups / nombre, sep="\t", header=None,
                     skiprows=1 if LOOKUPS[nombre] else 0, names=["codigo", "descripcion"],
                     dtype=str, keep_default_na=False, encoding="latin-1")
    return dict(df.values)


def _leer_zip(ruta: Path) -> pa.Table:
    tipos = {c: TIPOS_NUMERICOS.get(c, pa.string()) for c in COLUMNAS_EXPORT}
    with zipfile.ZipFile(ruta) as z:
        datos = z.read(z.namelist()[0])
    return pacsv.read_csv(
        io.BytesIO(datos),
        read_options=pacsv.ReadOptions(column_names=COLUMNAS_EXPORT),
        # GDELT no usa comillas: un " dentro de una URL no debe abrir un campo entrecomillado
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types=tipos, null_values=[""],
                                             strings_can_be_null=True),
    )


def leer_eventos(cfg: Config) -> pd.DataFrame:
    """Concatena todos los ZIP crudos de la ventana (61 columnas tipadas).

    Si un GlobalEventID aparece en más de un archivo, se conserva la última versión
    (la de DATEADDED más reciente)."""
    archivos = sorted(cfg.dir_gdelt.glob("*.export.CSV.zip"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos crudos en {cfg.dir_gdelt}. Corré descargar_ventana().")
    with ThreadPoolExecutor(max_workers=8) as ex:
        tabla = pa.concat_tables(list(ex.map(_leer_zip, archivos)))
    df = tabla.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype(), pa.int8(): pd.Int8Dtype()}.get)
    df = (df.sort_values("DATEADDED", kind="stable")
            .drop_duplicates("GlobalEventID", keep="last")
            .reset_index(drop=True))
    return df
