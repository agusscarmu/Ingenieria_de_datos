"""Extracción: baja SHDB-AF de PhysioNet a data/raw/ y lo lee (CSV y marcas de los Holter)."""
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import config

ESPERA_S = 60    # si la conexión queda muda tanto tiempo, la descarga falla en vez de colgarse
INTENTOS = 3     # intentos por archivo: un corte de red no frena el cuaderno entero
PAUSA_S = 5      # espera entre un intento y el siguiente (las pruebas la ponen en 0)
AVISO_CADA = 40  # cada cuántos archivos se avisa el avance, para que la celda no quede muda varios minutos


@dataclass(frozen=True)
class Holter:
    """Un Holter ya leído: sólo lo que hace falta de él, sin la señal."""
    registro: str                          # "001", nombre de sus archivos
    fs: int                                # Hz
    n_muestras: int                        # largo de la grabación
    qrs: np.ndarray                        # muestra de cada latido (int64)
    atr_muestras: np.ndarray | None        # muestras del .atr (int64); None si no está anotado
    atr_ritmos: list[str] | None           # aux_note tal cual: "(AFIB", "(N", ... y "" donde no hay marca


def _bajar(url: str, destino: Path) -> None:
    """Baja url a destino. Escribe a <destino>.parcial y renombra al terminar, así una
    descarga cortada no deja un archivo a medias que después se tome por completo.
    Si el servidor cierra antes de mandar todo lo que anunció (Content-Length), read()
    devuelve b"" en vez de fallar: por eso se compara el largo antes de renombrar."""
    parcial = destino.with_name(destino.name + ".parcial")
    try:
        with urllib.request.urlopen(url, timeout=ESPERA_S) as respuesta, open(parcial, "wb") as f:
            shutil.copyfileobj(respuesta, f)
            esperado = respuesta.headers.get("Content-Length")  # puede faltar (envío en trozos)
            if esperado is not None and f.tell() != int(esperado):
                raise OSError(f"descarga incompleta: {f.tell()} de {esperado} bytes de {url}")
        parcial.replace(destino)
    except BaseException:
        parcial.unlink(missing_ok=True)
        raise


def _bajar_si_falta(nombre: str, destino: Path) -> Path:
    """Baja destino/nombre si no está o está vacío (una descarga cortada puede dejar uno vacío).
    Si la descarga falla (OSError: la red se cae, el servidor corta), espera PAUSA_S y la reintenta,
    hasta INTENTOS veces; si falla la última, levanta el error."""
    ruta = destino / nombre
    if not (ruta.exists() and ruta.stat().st_size > 0):
        destino.mkdir(parents=True, exist_ok=True)
        for intento in range(1, INTENTOS + 1):
            try:
                _bajar(config.SHDB_AF_URL + nombre, ruta)
                break
            except OSError:
                if intento == INTENTOS:
                    raise
                time.sleep(PAUSA_S)
    return ruta


def descargar_csv(destino: Path = config.RAW) -> Path:
    """Baja AdditionalData.csv si no está y devuelve su ruta."""
    return _bajar_si_falta("AdditionalData.csv", destino)


def descargar_holters(registros: list[str], anotados: set[str], destino: Path = config.RAW) -> None:
    """Baja .hea y .qrs de cada registro, y .atr sólo de los anotados. No baja la señal (.dat).
    Cada AVISO_CADA archivos, y al terminar, avisa cuántos están listos (bajados o que ya estaban)."""
    nombres = [f"{registro}.{extension}" for registro in registros
               for extension in ["hea", "qrs"] + (["atr"] if registro in anotados else [])]
    for i, nombre in enumerate(nombres, start=1):
        _bajar_si_falta(nombre, destino)
        if i % AVISO_CADA == 0 or i == len(nombres):
            print(f"{i} de {len(nombres)} archivos listos")


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee de AdditionalData.csv sólo las columnas de config.COLUMNAS_CSV, ya con su tipo.
    Los faltantes de talla y peso quedan NaN: la limpieza viene después."""
    columnas = list(config.COLUMNAS_CSV)
    df = pd.read_csv(ruta, usecols=columnas, dtype=config.COLUMNAS_CSV)
    return df[columnas]  # read_csv las deja en el orden del archivo


def leer_holter(registro: str, anotado: bool, carpeta: Path = config.RAW) -> Holter:
    """Lee de la carpeta local el .hea, el .qrs y, si está anotado, el .atr de un Holter."""
    base = str(carpeta / registro)
    encabezado = wfdb.rdheader(base, pn_dir=None)
    qrs = wfdb.rdann(base, "qrs", pn_dir=None)
    atr = wfdb.rdann(base, "atr", pn_dir=None) if anotado else None
    return Holter(
        registro=registro,
        fs=int(encabezado.fs),
        n_muestras=int(encabezado.sig_len),
        qrs=np.asarray(qrs.sample, dtype=np.int64),
        atr_muestras=None if atr is None else np.asarray(atr.sample, dtype=np.int64),
        atr_ritmos=None if atr is None else list(atr.aux_note),
    )
