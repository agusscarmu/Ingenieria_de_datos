"""Rutas y constantes del TP: el resto del código las toma de acá."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DATA = RAIZ / "data"
RAW = DATA / "raw"              # SHDB-AF tal como se baja de PhysioNet
DB = DATA / "db"
SQL = RAIZ / "sql"              # DDL de las dos bases

TRANSACCIONAL = DB / "transaccional.sqlite"
ALMACEN = DB / "almacen.sqlite"

# SHDB-AF v1.0.1, PhysioNet, licencia ODC-By 1.0
SHDB_AF_URL = "https://physionet.org/files/shdb-af/1.0.1/"

# Ventanas y franjas del día
VENTANA_S = 30                  # largo de cada ventana de análisis (s)
NOCHE_DESDE = 23                # noche: hora entera >= 23 o < 7 (el resto es día)
NOCHE_HASTA = 7

# RR de cada ventana
MIN_RR = 5                      # mínimo de RR por ventana para calcular las 17 (el de la tesis)
RR_MIN_MS = 240                 # RR atípico: menos de esto (más de 250 lpm); se cuenta y se conserva
RR_MAX_MS = 2000                # RR atípico: más de esto (menos de 30 lpm); se cuenta y se conserva

# Columnas de AdditionalData.csv que se leen (el resto no se carga) y su tipo en pandas
COLUMNAS_CSV = {
    "Data_ID": "int16",                  # id del Holter
    "Subject_ID": "int32",               # id del paciente
    "Annotated": "bool",                 # tiene .atr
    "AF_Type": "category",
    "Age_at_Holter": "int16",
    "Sex": "category",
    "Height": "float32",                 # m; faltante = NaN
    "Weight": "float32",                 # kg; faltante = NaN
    "Holter_start_time": "string",       # hh:mm AM/PM (se pasa a 24 h)
    "Holter_recording_length": "string", # sólo para validar
}

TIPOS_FA = {"PAF": "FA paroxística", "PerAF": "FA persistente", "non-AF": "Sin FA"}

# Ritmos que marca el .atr (aux_note, sin el paréntesis)
RITMOS = {
    "AFIB": "Fibrilación auricular",
    "N": "Normal: ritmo no marcado como otra cosa",
    "AFL": "Aleteo auricular",
    "AT": "Taquicardia auricular",
    "AB": "Bigeminia auricular",
    "PAT": "Taquicardia auricular paroxística",
    "NOD": "Ritmo nodal",
}
