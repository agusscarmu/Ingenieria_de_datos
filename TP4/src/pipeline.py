"""Pipeline de datos: crudos (data/raw) -> One Big Table (data/processed).

Uso, desde la raíz del TP4:
    python -m src.pipeline
"""
from __future__ import annotations

import time

from .config import Config, cargar_config
from .ingesta import construir_manifiesto, descargar_lookups, descargar_ventana, leer_eventos
from .obt import construir_obt


def ejecutar(cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or cargar_config()
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()

    log(f"[1/3] Manifiesto de la ventana {cfg.fecha_inicio} a {cfg.fecha_fin} (masterfilelist.txt)")
    manifiesto = construir_manifiesto(cfg)
    log(f"      {len(manifiesto)} archivos export.CSV.zip ({manifiesto['bytes'].sum() / 1e6:.1f} MB)")

    log("[2/3] Descargando a data/raw (con verificación MD5)")
    reporte = descargar_ventana(cfg, manifiesto)
    descargar_lookups(cfg)
    log("      " + ", ".join(f"{k}: {v}" for k, v in reporte["estado"].value_counts().items()))

    log("[3/3] Parseando y construyendo la One Big Table")
    eventos = leer_eventos(cfg)
    obt = construir_obt(cfg, eventos)
    obt.to_parquet(cfg.ruta("obt"), index=False)
    log(f"      OBT: {obt.shape[0]:,} eventos x {obt.shape[1]} columnas -> {cfg.rutas['obt']}")

    log(f"Listo en {time.perf_counter() - t0:.1f} s")
    return {"manifiesto": manifiesto, "reporte_descarga": reporte, "obt": obt}


if __name__ == "__main__":
    ejecutar()
