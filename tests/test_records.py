import math

import pytest

from robustness_gap import records
from robustness_gap.records import Status


def test_nan_and_inf_rejected(tmp_path):
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            records.atomic_write_json(tmp_path / "r.json", {"U": bad})
    assert not (tmp_path / "r.json").exists()
    assert not list(tmp_path.glob(".*tmp"))


def test_status_transitions():
    records.check_status("attack", Status.NO_VALID_CANDIDATE)
    with pytest.raises(ValueError):
        records.check_status("certificate", Status.TIMEOUT_WITH_BOUNDS)


def test_stale_record_detected(tmp_path):
    p = tmp_path / "inst.json"
    records.atomic_write_json(p, records.make_record("instance", {"L": 0.1}, input_hash="aaa"))
    assert records.load_if_current(p, "aaa")["L"] == 0.1
    with pytest.raises(records.StaleRecordError):
        records.load_if_current(p, "bbb")
    assert records.load_if_current(tmp_path / "missing.json", "aaa") is None


def test_write_new_refuses_overwrite(tmp_path):
    p = tmp_path / "panel.json"
    records.write_new_json(p, {"ids": [1, 2]})
    records.write_new_json(p, {"ids": [1, 2]})
    with pytest.raises(FileExistsError):
        records.write_new_json(p, {"ids": [3]})


def test_hash_is_order_independent():
    assert records.sha256_json({"a": 1, "b": 2}) == records.sha256_json({"b": 2, "a": 1})
