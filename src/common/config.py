"""Chargement de la configuration YAML (avec héritage via la clé `extends`)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "params.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _merge(load_yaml(path.parent / parent), data)
    return data


@dataclass
class Config:
    raw: dict
    source: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, name: str) -> Path:
        """Chemin absolu d'une entrée de la section `paths`."""
        return (ROOT / self.raw["paths"][name]).resolve()

    def interim(self, filename: str) -> Path:
        return self.path("interim") / filename

    def processed(self, filename: str) -> Path:
        return self.path("processed") / filename

    def report(self, filename: str) -> Path:
        return self.path("reports") / filename


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return Config(load_yaml(p), p)
