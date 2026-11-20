"""Base transaccional (SQLite, 3FN): se crea desde sql/transaccional.sql y se carga con las tablas que deja la transformación."""
import sqlite3
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

import config

COLUMNAS_PACIENTE = ["paciente_id", "sexo"]
COLUMNAS_HOLTER = ["holter_id", "paciente_id", "tipo_fa", "anotado", "edad", "talla_m", "peso_kg", "hora_inicio",
                   "fs_hz", "n_muestras"]
COLUMNAS_LATIDO = ["holter_id", "n", "muestra"]


def crear(ruta: Path = config.TRANSACCIONAL) -> sqlite3.Connection:
    """Crea la base de cero (borra la anterior si existe) y devuelve la conexión abierta, con las claves foráneas activas.

    Rehacerla en cada corrida evita filas duplicadas y errores de clave al re-ejecutar el cuaderno. Si la corrida
    anterior dejó la conexión abierta no importa: el archivo viejo se desvincula y esa conexión sigue con el suyo.
    """
    ddl = (config.SQL / "transaccional.sql").read_text(encoding="utf-8")
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    for sufijo in ("", "-journal", "-wal", "-shm"):     # un diario viejo sobre la base nueva la rompería
        Path(f"{ruta}{sufijo}").unlink(missing_ok=True)
    con = sqlite3.connect(ruta)
    con.execute("PRAGMA foreign_keys = ON")             # SQLite las trae apagadas, y se prenden por conexión
    con.executescript(ddl)
    return con


def _insertar(con: sqlite3.Connection, tabla: str, columnas: list[str], filas: Iterable[tuple]) -> None:
    """INSERT de `filas` (una tupla por fila, en el orden de `columnas`) en `tabla`. No hace commit: lo hace quien llama."""
    huecos = ", ".join("?" * len(columnas))
    con.executemany(f"INSERT INTO {tabla} ({', '.join(columnas)}) VALUES ({huecos})", filas)


def _filas(tabla: pd.DataFrame, columnas: list[str]) -> list[tuple]:
    """Filas de `tabla` en el orden de `columnas`, con tipos de Python (sqlite3 no acepta los de numpy) y faltantes como None."""
    elegida = tabla[columnas]
    return list(elegida.astype(object).where(elegida.notna(), None).itertuples(index=False, name=None))


def cargar_catalogos(con: sqlite3.Connection) -> None:
    """Llena `tipo_fa` y `ritmo` con los códigos y descripciones de config."""
    with con:
        _insertar(con, "tipo_fa", ["tipo_fa", "descripcion"], config.TIPOS_FA.items())
        _insertar(con, "ritmo", ["ritmo", "descripcion"], config.RITMOS.items())


def cargar_clinicas(con: sqlite3.Connection, paciente: pd.DataFrame, holter: pd.DataFrame) -> None:
    """Carga `paciente` y `holter` (las que arma `transformacion.tablas_clinicas`) en una sola transacción.

    Talla y peso faltantes quedan NULL, sin imputar, y los booleanos se guardan como 0/1. Los catálogos tienen que
    estar cargados.
    """
    with con:
        _insertar(con, "paciente", COLUMNAS_PACIENTE, _filas(paciente, COLUMNAS_PACIENTE))
        _insertar(con, "holter", COLUMNAS_HOLTER, _filas(holter, COLUMNAS_HOLTER))


def cargar_latidos(con: sqlite3.Connection, holter_id: int, muestras: np.ndarray) -> None:
    """Carga los latidos de un Holter: `muestras`, la posición de cada latido. `n` es el orden del latido, desde 0.

    Va en una sola transacción: si algo falla, el Holter no queda cargado a medias. El Holter tiene que estar en
    `holter`.
    """
    filas = zip(repeat(int(holter_id)), range(len(muestras)), muestras.tolist())    # zip arma las tuplas de a una
    with con:
        _insertar(con, "latido", COLUMNAS_LATIDO, filas)


def cargar_episodios(con: sqlite3.Connection, holter_id: int, episodios: list[tuple[int, str]]) -> None:
    """Carga los episodios de ritmo de un Holter, `(latido_inicio, ritmo)` como los da `transformacion.episodios`.

    Sus latidos tienen que estar cargados: cada episodio arranca en uno.
    """
    with con:
        _insertar(con, "episodio_ritmo", ["holter_id", "latido_inicio", "ritmo"],
                  [(int(holter_id), int(inicio), ritmo) for inicio, ritmo in episodios])


def contar(con: sqlite3.Connection) -> dict[str, int]:
    """Filas de cada tabla, en el orden en que se crearon."""
    tablas = [nombre for (nombre,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    return {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tablas}
