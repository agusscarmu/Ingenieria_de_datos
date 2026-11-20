"""Ventanas de 30 s y RR de cada una.

Las ventanas se cortan cada `VENTANA_S` segundos desde la primera muestra del registro, y sólo cuentan las enteras. Un
latido es de la ventana `muestra // (VENTANA_S · fs)`. Un RR es de una ventana sólo si sus dos latidos caen en ella: el
que cruza un borde no es de ninguna, y tampoco los de la ventana incompleta del final.
"""
from collections.abc import Iterator

import numpy as np

import config


def n_ventanas(n_muestras: int, fs: int) -> int:
    """Cuántas ventanas de `VENTANA_S` caben enteras en un registro de `n_muestras` muestras a `fs` Hz."""
    return n_muestras // (config.VENTANA_S * fs)


def cortes(muestras: np.ndarray, fs: int, n_muestras: int) -> np.ndarray:
    """Posición del primer latido de cada ventana entera, más una de cierre: los latidos de la ventana `w` son
    `cortes[w]:cortes[w + 1]`. Los de la ventana incompleta del final quedan afuera. `muestras` va en orden."""
    bordes = np.arange(n_ventanas(n_muestras, fs) + 1, dtype=np.int64) * (config.VENTANA_S * fs)
    return np.searchsorted(muestras, bordes, side="left")


def rr_de_ventanas(muestras: np.ndarray, fs: int, n_muestras: int) -> Iterator[tuple[int, np.ndarray]]:
    """Da `(n_ventana, rr_ms)` de cada ventana entera, en orden, también de las vacías.

    `muestras` son las posiciones de los latidos, en orden. `rr_ms` (float64) son los RR entre los latidos de la
    ventana, en ms; una ventana sin latidos o con uno solo da un arreglo vacío.
    """
    muestras = np.asarray(muestras, dtype=np.int64)
    c = cortes(muestras, fs, n_muestras).tolist()
    for n_ventana, (desde, hasta) in enumerate(zip(c[:-1], c[1:])):
        yield n_ventana, np.diff(muestras[desde:hasta]) * 1000 / fs


def contiguos(n: int) -> np.ndarray:
    """La contigüidad que piden las 17 para una serie cruda de `n` RR: el primero `False` y los demás `True`, porque
    todos los RR de una ventana son vecinos (cada uno empieza donde terminó el anterior)."""
    c = np.ones(n, dtype=bool)
    c[:1] = False
    return c
