"""Carga de la configuración del proyecto (config.toml en la raíz del TP)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SECCIONES = ("ingesta", "rutas", "limpieza", "matriz", "acp", "atipicos")


@dataclass(frozen=True)
class Config:
    raw: dict
    ingesta: dict
    rutas: dict
    limpieza: dict
    matriz: dict
    acp: dict
    atipicos: dict

    @property
    def fecha_fin(self) -> date:
        return date.fromisoformat(self.ingesta["fecha_fin"])

    @property
    def fecha_inicio(self) -> date:
        return self.fecha_fin - timedelta(days=self.ingesta["dias"] - 1)

    @property
    def ventana(self) -> tuple[datetime, datetime]:
        """[inicio, fin) en UTC: desde las 00:00 del primer día hasta las 00:00 del día siguiente al último."""
        inicio = datetime.combine(self.fecha_inicio, datetime.min.time())
        return inicio, inicio + timedelta(days=self.ingesta["dias"])

    def ruta(self, clave: str) -> Path:
        return RAIZ / self.rutas[clave]

    @property
    def dir_gdelt(self) -> Path:
        """Carpeta de los export.CSV.zip crudos de la ventana."""
        return self.ruta("raw") / "gdelt" / f"{self.fecha_inicio:%Y%m%d}_{self.fecha_fin:%Y%m%d}"

    @property
    def dir_lookups(self) -> Path:
        return self.ruta("raw") / "lookups"


def cargar_config(ruta: Path | str | None = None) -> Config:
    ruta = Path(ruta) if ruta else RAIZ / "config.toml"
    with open(ruta, "rb") as f:
        d = tomllib.load(f)
    return Config(raw=d, **{k: d[k] for k in SECCIONES})
