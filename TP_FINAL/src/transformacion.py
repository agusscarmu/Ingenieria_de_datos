"""Transformación (ETL 1): limpia el CSV clínico, cuenta los RR atípicos y saca los episodios de ritmo de las marcas del
.atr."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
import ventanas
from extraccion import Holter

_DURACION = re.compile(r"(\d+):(\d+):(\d+)")           # h:m:s, sin ceros a la izquierda
_HORA_12 = re.compile(r"(\d{1,2}):(\d{2}) (AM|PM)")    # "04:26 PM"


def cohorte(csv: pd.DataFrame) -> pd.DataFrame:
    """Holter que entran al análisis: los anotados (con .atr) y los sin FA, que son todo ritmo N."""
    return csv[_en_cohorte(csv)].reset_index(drop=True)


def excluidos(csv: pd.DataFrame) -> pd.DataFrame:
    """Los que quedan afuera: con FA y sin .atr, así que no se sabe qué ventanas están en FA."""
    return csv[~_en_cohorte(csv)].reset_index(drop=True)


def _en_cohorte(csv: pd.DataFrame) -> pd.Series:
    """True en los Holter de la cohorte: los anotados, más los sin anotar que son sin FA (non-AF)."""
    return csv["Annotated"] | (csv["AF_Type"] == "non-AF")


def duracion_s(texto: str) -> int:
    """Segundos de un "h:m:s" del CSV. Acepta el formato irregular ("24:4:59", "8:59:59")."""
    partes = _DURACION.fullmatch(texto.strip())
    if partes is None:
        raise ValueError(f"duración con formato inesperado: {texto!r}")
    horas, minutos, segundos = (int(p) for p in partes.groups())
    return horas * 3600 + minutos * 60 + segundos


def hora_24(texto: str) -> str:
    """Pasa "hh:mm AM/PM" del CSV a "HH:MM" de 24 horas: "04:26 PM" -> "16:26"."""
    partes = _HORA_12.fullmatch(texto.strip().upper())
    if partes is None or not (1 <= int(partes[1]) <= 12 and int(partes[2]) < 60):
        raise ValueError(f"hora con formato inesperado (se espera hh:mm AM/PM): {texto!r}")
    hora = int(partes[1]) % 12 + (12 if partes[3] == "PM" else 0)  # 12 AM = 0, 12 PM = 12
    return f"{hora:02d}:{partes[2]}"


def edad_75_inconsistente(ruta_csv: Path) -> pd.DataFrame:
    """Filas donde Age_75_or_Older no coincide con la edad (75 ya cuenta como "75 o más").
    Lee sólo esas tres columnas: sirve de ejemplo de por qué no se guardan datos derivables."""
    columnas = ["Data_ID", "Age_at_Holter", "Age_75_or_Older"]
    tipos = {"Data_ID": config.COLUMNAS_CSV["Data_ID"],
             "Age_at_Holter": config.COLUMNAS_CSV["Age_at_Holter"],
             "Age_75_or_Older": "bool"}
    df = pd.read_csv(ruta_csv, usecols=columnas, dtype=tipos)[columnas]
    return df[df["Age_75_or_Older"] != (df["Age_at_Holter"] >= 75)].reset_index(drop=True)


def comparar_duraciones(cohorte: pd.DataFrame, holters: dict[int, Holter]) -> pd.DataFrame:
    """Duración del CSV contra la del .hea, en segundos enteros, de cada Holter de la cohorte.
    diferencia_s = hea_s - csv_s. Sólo se muestra: la duración que se guarda es la del .hea."""
    hea_s = []
    for holter_id in cohorte["Data_ID"]:
        h = holters[int(holter_id)]
        hea_s.append(h.n_muestras // h.fs)  # el CSV trunca la fracción de segundo
    comparacion = pd.DataFrame({
        "holter_id": cohorte["Data_ID"].astype("int64").to_numpy(),
        "csv_s": [duracion_s(t) for t in cohorte["Holter_recording_length"]],
        "hea_s": hea_s,
    })
    comparacion["diferencia_s"] = comparacion["hea_s"] - comparacion["csv_s"]
    return comparacion


def rr_atipicos(holters: dict[int, Holter]) -> pd.DataFrame:
    """Una fila: cuántos RR hay en las ventanas de 30 s (los que se usan) y fuera de ellas (cruzan un borde o están en
    la ventana incompleta del final), cuántos de los de las ventanas son atípicos (menos de `RR_MIN_MS` o más de
    `RR_MAX_MS`), qué % son y el RR más largo de las ventanas. Sólo cuenta: los atípicos se conservan."""
    rr = np.concatenate([rr_ms for h in holters.values()
                         for _, rr_ms in ventanas.rr_de_ventanas(h.qrs, h.fs, h.n_muestras)])
    todos = sum(len(h.qrs) - 1 for h in holters.values() if len(h.qrs) > 0)
    cortos = int((rr < config.RR_MIN_MS).sum())
    largos = int((rr > config.RR_MAX_MS).sum())
    return pd.DataFrame({"RR en ventanas": [rr.size],
                         "RR fuera de ventanas": [todos - rr.size],
                         f"< {config.RR_MIN_MS} ms": [cortos],
                         f"> {config.RR_MAX_MS} ms": [largos],
                         f"< {config.RR_MIN_MS} ms, % de los RR en ventanas": [100 * cortos / rr.size],
                         f"> {config.RR_MAX_MS} ms, % de los RR en ventanas": [100 * largos / rr.size],
                         "RR más largo en ventanas (ms)": [rr.max()]}, index=["RR"])


def tablas_clinicas(cohorte: pd.DataFrame, holters: dict[int, Holter]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte la cohorte en las tablas paciente y holter de la base transaccional (3FN):
    el sexo es del paciente, y fs y largo salen del .hea. Talla y peso quedan nulos donde faltan,
    sin imputar. Si un paciente aparece con dos sexos, frena: no se sabe cuál es el cierto."""
    c = cohorte.reset_index(drop=True)
    sexos = c.groupby("Subject_ID")["Sex"].nunique()
    if (sexos > 1).any():
        raise ValueError(f"pacientes con más de un sexo en el CSV: {sexos[sexos > 1].index.tolist()}")

    paciente = (c[["Subject_ID", "Sex"]].drop_duplicates("Subject_ID").sort_values("Subject_ID")
                .rename(columns={"Subject_ID": "paciente_id", "Sex": "sexo"}).reset_index(drop=True))
    paciente = paciente.astype({"paciente_id": "int64", "sexo": "str"})

    de_holter = [holters[int(i)] for i in c["Data_ID"]]
    holter = pd.DataFrame({
        "holter_id": c["Data_ID"].astype("int64"),
        "paciente_id": c["Subject_ID"].astype("int64"),
        "tipo_fa": c["AF_Type"].astype("str"),
        "anotado": c["Annotated"].astype("bool"),
        "edad": c["Age_at_Holter"].astype("int64"),
        # float64 redondeado: sin el ruido de float32 (1.73 no queda 1.7300000190734863 en la base)
        "talla_m": c["Height"].astype("float64").round(4),
        "peso_kg": c["Weight"].astype("float64").round(4),
        "hora_inicio": [hora_24(t) for t in c["Holter_start_time"]],
        "fs_hz": [h.fs for h in de_holter],
        "n_muestras": [h.n_muestras for h in de_holter],
    })
    return paciente, holter


def episodios(holter: Holter) -> list[tuple[int, str]]:
    """Episodios de ritmo de un Holter: (latido_inicio, código sin paréntesis), uno por cada
    marca del .atr. Sin .atr, no hay episodios. Frena con ValueError si el .atr no cumple lo
    medido en SHDB-AF: mismas muestras que el .qrs, marca en el primer latido y sólo los 7 ritmos."""
    if holter.atr_muestras is None:
        return []
    if not np.array_equal(holter.atr_muestras, holter.qrs):
        raise ValueError(f"Holter {holter.registro}: el .atr no marca las mismas muestras que el .qrs")
    notas = holter.atr_ritmos
    if not notas or not notas[0]:
        raise ValueError(f"Holter {holter.registro}: la primera marca de ritmo no está en el primer latido")

    resultado = []
    for latido, nota in enumerate(notas):
        if nota == "":  # el ritmo se marca sólo donde cambia
            continue
        codigo = nota.removeprefix("(")
        if not nota.startswith("(") or codigo not in config.RITMOS:
            raise ValueError(f"Holter {holter.registro}: ritmo desconocido {nota!r} en el latido {latido}")
        resultado.append((latido, codigo))
    return resultado
