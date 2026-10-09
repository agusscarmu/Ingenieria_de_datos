"""Pipeline de datos: crudos (data/raw) -> almacén + matriz (data/processed).

Uso, desde la raíz del TP3:
    python -m src.pipeline
"""
from __future__ import annotations

import time

from .almacen import construir_almacen, leer_vista_analisis
from .config import Config, cargar_config
from .ingesta import descargar_dia, descargar_lookups, leer_eventos
from .matriz import construir_matriz


def ejecutar(cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or cargar_config()
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()

    log(f"[1/4] Descargando export.CSV del {cfg.fecha} -> {cfg.dir_gdelt.relative_to(cfg.ruta('raw').parent.parent)}")
    reporte = descargar_dia(cfg)
    descargar_lookups(cfg)
    log("      " + ", ".join(f"{k}: {v}" for k, v in reporte["estado"].value_counts().items()))

    log("[2/4] Parseando archivos crudos")
    eventos = leer_eventos(cfg)
    log(f"      {len(eventos):,} eventos únicos")

    log("[3/4] Construyendo almacén de datos")
    filas = construir_almacen(cfg, eventos)
    for tabla, n in filas.items():
        log(f"      {tabla:<24} {n:>9,}")
    vista = leer_vista_analisis(cfg)
    vista.to_parquet(cfg.ruta("eventos"), index=False)

    log("[4/4] Construyendo matriz de datos")
    matriz = construir_matriz(vista, cfg)
    matriz.to_parquet(cfg.ruta("matriz"))
    log(f"      matriz: {matriz.shape[0]:,} filas x {matriz.shape[1]} columnas")

    log(f"Listo en {time.perf_counter() - t0:.1f} s")
    return {"reporte_descarga": reporte, "filas_almacen": filas, "eventos": vista, "matriz": matriz}


if __name__ == "__main__":
    ejecutar()
