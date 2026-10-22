from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import records
from .config import load_config

SETTINGS_FILE = "lab-settings.json"
SMOKE_OVERRIDE = "configs/overrides/smoke.yaml"


@dataclass
class LabSettings:
    config: str = "configs/pilot.yaml"
    smoke: bool = False
    machine: str = ""
    coordinator: str = ""
    access_code: str = ""
    workers: int = 1

    def overrides(self, root: Path) -> list[Path]:
        return [root / SMOKE_OVERRIDE] if self.smoke else []

    def load_config(self, root: Path) -> tuple[dict, dict]:
        return load_config(root / self.config, self.overrides(root))


def load_settings(root: str | Path) -> LabSettings | None:
    path = Path(root) / SETTINGS_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(LabSettings)}
    return LabSettings(**{k: v for k, v in data.items() if k in known})


def save_settings(root: str | Path, settings: LabSettings) -> None:
    records.atomic_write_json(Path(root) / SETTINGS_FILE, asdict(settings))
