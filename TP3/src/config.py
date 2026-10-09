"""Carga de la configuración del proyecto (config.toml en la raíz del TP)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    raw: dict
    ingesta: dict
    rutas: dict
    matriz: dict
    multivariante: dict
    acp: dict

    @property
    def fecha(self) -> date:
        return date.fromisoformat(self.ingesta["fecha"])

    def ruta(self, clave: str) -> Path:
        return RAIZ / self.rutas[clave]

    @property
    def dir_gdelt(self) -> Path:
        """Carpeta de los export.CSV.zip crudos del día analizado."""
        return self.ruta("raw") / "gdelt" / self.fecha.strftime("%Y%m%d")

    @property
    def dir_lookups(self) -> Path:
        return self.ruta("raw") / "lookups"


def cargar_config(ruta: Path | str | None = None) -> Config:
    ruta = Path(ruta) if ruta else RAIZ / "config.toml"
    with open(ruta, "rb") as f:
        d = tomllib.load(f)
    return Config(raw=d, **{k: d[k] for k in ("ingesta", "rutas", "matriz", "multivariante", "acp")})
