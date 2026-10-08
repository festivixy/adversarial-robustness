from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .records import sha256_json

STAGES = ("smoke", "validation", "pilot", "main")
REQUIRED_SECTIONS = (
    "experiment", "domain", "dataset", "model", "training", "panel",
    "target", "certificate", "attack", "reference", "tolerances", "diagnostics",
)
OPTIONAL_NULLS = frozenset({
    "target.margin", "freeze.date", "freeze.config_hash",
    "freeze.readiness_review", "freeze.teacher_review",
})


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str | Path] = ()) -> tuple[dict, dict]:
    authored = load_yaml(path)
    merged = authored
    for o in overrides:
        merged = deep_merge(merged, load_yaml(o))
    if overrides and merged["experiment"].get("stage") != "smoke":
        raise ConfigError("overrides are only for smoke runs and must set experiment.stage: smoke")
    validate_structure(merged)
    return authored, merged


def validate_structure(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"missing sections: {missing}")
    stage = cfg["experiment"].get("stage")
    if stage not in STAGES:
        raise ConfigError(f"experiment.stage must be one of {STAGES}, got {stage!r}")
    exp_id = cfg["experiment"].get("id")
    if not exp_id:
        raise ConfigError("experiment.id is required")
    if (stage == "smoke") != ("smoke" in exp_id):
        raise ConfigError("an experiment id contains 'smoke' if and only if its stage is smoke")
    d = cfg["dataset"]
    if d["n_train"] + d["n_val"] + d["n_test"] != d["n_total"]:
        raise ConfigError("dataset split sizes do not sum to n_total")
    if d["n_total"] % 2:
        raise ConfigError("balanced two-class dataset needs an even n_total")


def unresolved_settings(cfg: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in cfg.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(unresolved_settings(v, f"{path}."))
        elif v is None and path not in OPTIONAL_NULLS:
            out.append(path)
    return out


def scientific_section(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in ("experiment", "freeze")}


def scientific_hash(cfg: dict) -> str:
    return sha256_json(scientific_section(cfg))


def check_main_run_allowed(cfg: dict, main_flag: bool) -> None:
    if cfg["experiment"]["stage"] != "main":
        if main_flag:
            raise ConfigError("--main-run given but config stage is not 'main'")
        return
    if not main_flag:
        raise ConfigError("main-stage config requires the explicit --main-run flag")
    unresolved = unresolved_settings(cfg)
    if unresolved:
        raise ConfigError(f"main run blocked; unresolved settings: {unresolved}")
    freeze = cfg.get("freeze") or {}
    if not freeze.get("frozen") or not freeze.get("readiness_review"):
        raise ConfigError("main run blocked: freeze.frozen and freeze.readiness_review must be recorded")
    if freeze.get("config_hash") != scientific_hash(cfg):
        raise ConfigError("main run blocked: freeze.config_hash does not match the current config")


def get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node
