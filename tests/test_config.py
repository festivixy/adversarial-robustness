import copy

import pytest

from robustness_gap import config as C

from conftest import REPO


def test_smoke_override_labels_stage():
    _, cfg = C.load_config(REPO / "configs/pilot.yaml", [REPO / "configs/overrides/smoke.yaml"])
    assert cfg["experiment"]["stage"] == "smoke" and "smoke" in cfg["experiment"]["id"]


def test_smoke_changes_scientific_hash(pilot_cfg):
    _, smoke = C.load_config(REPO / "configs/pilot.yaml", [REPO / "configs/overrides/smoke.yaml"])
    assert C.scientific_hash(smoke) != C.scientific_hash(pilot_cfg)


def test_non_smoke_cannot_use_smoke_id(pilot_cfg):
    cfg = copy.deepcopy(pilot_cfg)
    cfg["experiment"]["id"] = "smoke-sneaky"
    with pytest.raises(C.ConfigError):
        C.validate_structure(cfg)


def test_main_config_is_blocked_until_frozen():
    _, cfg = C.load_config(REPO / "configs/main.yaml")
    with pytest.raises(C.ConfigError, match="--main-run"):
        C.check_main_run_allowed(cfg, main_flag=False)
    with pytest.raises(C.ConfigError, match="unresolved"):
        C.check_main_run_allowed(cfg, main_flag=True)


def test_main_guard_passes_only_with_matching_freeze():
    _, cfg = C.load_config(REPO / "configs/main.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["target"]["policy"] = "score_boundary"
    for k in ("certificate.numerical_margin", "attack.implementation", "reference.mip_rel_gap",
              "reference.primal_feasibility_tolerance", "diagnostics.relu_stability_radius"):
        sec, key = k.split(".")
        cfg[sec][key] = "x"
    cfg["freeze"].update(frozen=True, readiness_review="r.md", config_hash="wrong")
    with pytest.raises(C.ConfigError, match="config_hash"):
        C.check_main_run_allowed(cfg, True)
    cfg["freeze"]["config_hash"] = C.scientific_hash(cfg)
    C.check_main_run_allowed(cfg, True)


def test_pilot_rejects_main_flag(pilot_cfg):
    with pytest.raises(C.ConfigError):
        C.check_main_run_allowed(pilot_cfg, main_flag=True)
