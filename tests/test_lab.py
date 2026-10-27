import json
import zipfile

import pytest

from robustness_gap import collection, lab, records
from robustness_gap.analyze import summarize_clean
from robustness_gap.cli import main
from robustness_gap.config import load_config, scientific_hash

from conftest import REPO


@pytest.fixture
def smoke_cfg():
    return load_config(REPO / "configs/pilot.yaml", [REPO / "configs/overrides/smoke.yaml"])[1]


def quiet(*_):
    pass


def test_jobs_follow_config_order(pilot_cfg):
    ids = [j.job_id for j in collection.all_jobs(pilot_cfg)]
    assert ids == ["mlp1-seed9001", "mlp1-seed9002", "mlp2-seed9001", "mlp2-seed9002",
                   "mlp3-seed9001", "mlp3-seed9002"]


@pytest.mark.parametrize("n", [1, 2, 4, 6, 10])
def test_assignment_covers_every_job_exactly_once(pilot_cfg, n):
    jobs = collection.all_jobs(pilot_cfg)
    assigned = [j for js in collection.assign(jobs, n).values() for j in js]
    assert sorted(j.job_id for j in assigned) == sorted(j.job_id for j in jobs)


def test_slot_selection(pilot_cfg):
    assert [j.job_id for j in collection.select_jobs(pilot_cfg, slot="2/4")] == ["mlp1-seed9002", "mlp3-seed9002"]
    for bad in ("0/4", "5/4", "two/4", "3"):
        with pytest.raises(collection.LabError):
            collection.parse_slot(bad)


def test_exactly_one_selection_mode(pilot_cfg):
    with pytest.raises(collection.LabError):
        collection.select_jobs(pilot_cfg)
    with pytest.raises(collection.LabError):
        collection.select_jobs(pilot_cfg, job_ids=["mlp1-seed9001"], everything=True)
    with pytest.raises(collection.LabError, match="unknown"):
        collection.select_jobs(pilot_cfg, job_ids=["mlp9-seed1"])


@pytest.mark.parametrize("name", ["", "lab 07", "../x", "a/b", "x" * 41])
def test_bad_machine_names_rejected(name):
    with pytest.raises(collection.LabError):
        collection.check_machine(name)


def test_work_is_resumable_and_logged(tmp_path, smoke_cfg):
    jobs = collection.select_jobs(smoke_cfg, job_ids=["mlp1-seed0"])
    first = lab.work(tmp_path, smoke_cfg, jobs, "pc-a", say=quiet)
    panel = smoke_cfg["panel"]["size"]
    assert first[0]["stages"]["clean"] == {"new": panel, "already_done": 0}
    assert "D001" in first[0]["stages"]["certificate"]
    second = lab.work(tmp_path, smoke_cfg, jobs, "pc-a", say=quiet)
    assert second[0]["stages"]["clean"] == {"new": 0, "already_done": panel}

    sessions = sorted((tmp_path / "runs/smoke-pilot-v0/machines/pc-a/sessions").glob("*.jsonl"))
    assert len(sessions) == 2
    events = [e["event"] for e in collection.read_jsonl(sessions[0])]
    assert events == ["session_start", "job_start", "job_done", "session_end"]
    model = records.read_json(tmp_path / "runs/smoke-pilot-v0/models/mlp1-seed0/record.json")
    assert model["machine"] == "pc-a" and model["code_hash"]


def test_parallel_workers(tmp_path, smoke_cfg):
    jobs = collection.select_jobs(smoke_cfg, everything=True)
    out = lab.work(tmp_path, smoke_cfg, jobs, "pc-a", workers=2, say=quiet)
    assert sorted(s["job"] for s in out) == sorted(j.job_id for j in jobs)
    assert all("error" not in s for s in out)


def test_failed_job_is_recorded_not_hidden(tmp_path, smoke_cfg, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(lab, "run_job", boom)
    out = lab.work(tmp_path, smoke_cfg, collection.select_jobs(smoke_cfg, everything=True)[:1], "pc-a", say=quiet)
    assert "disk full" in out[0]["error"]
    log = next((tmp_path / "runs/smoke-pilot-v0/machines/pc-a/sessions").glob("*.jsonl"))
    assert "job_failed" in [e["event"] for e in collection.read_jsonl(log)]


def two_machines(tmp_path, cfg):
    a, b = tmp_path / "a", tmp_path / "b"
    lab.work(a, cfg, collection.select_jobs(cfg, slot="1/2"), "pc-a", say=quiet)
    lab.work(b, cfg, collection.select_jobs(cfg, slot="2/2"), "pc-b", say=quiet)
    return a, b


def test_export_import_merges_machines(tmp_path, smoke_cfg):
    a, b = two_machines(tmp_path, smoke_cfg)
    report = collection.import_bundle(a, smoke_cfg, collection.export_bundle(b, smoke_cfg, "pc-b"))
    assert report["added"] > 0 and report["conflicts_kept_existing"] == []
    rows = collection.progress(a, smoke_cfg)["jobs"]
    assert all(r["trained"] == "success" and r["clean"] == smoke_cfg["panel"]["size"] for r in rows)
    assert {r["machine"] for r in rows} == {"pc-a", "pc-b"}
    assert summarize_clean(a, smoke_cfg)["complete"]

    again = collection.import_bundle(a, smoke_cfg, collection.export_bundle(b, smoke_cfg, "pc-b"))
    assert again["added"] == 0 and again["identical"] > 0


def test_grown_logs_extend_on_reimport(tmp_path, smoke_cfg):
    a, b = two_machines(tmp_path, smoke_cfg)
    collection.add_note(b, smoke_cfg, "pc-b", "first")
    collection.import_bundle(a, smoke_cfg, collection.export_bundle(b, smoke_cfg, "pc-b"))
    collection.add_note(b, smoke_cfg, "pc-b", "second")
    report = collection.import_bundle(a, smoke_cfg, collection.export_bundle(b, smoke_cfg, "pc-b"))
    assert report["extended_logs"] == 1
    assert [n["note"] for n in collection.all_notes(a, smoke_cfg)] == ["first", "second"]


def test_conflict_stops_import_before_writing(tmp_path, smoke_cfg):
    a, b = two_machines(tmp_path, smoke_cfg)
    lab.work(a, smoke_cfg, collection.select_jobs(smoke_cfg, slot="2/2"), "pc-a", say=quiet)
    before = sorted(p for p in a.rglob("*") if p.is_file())
    bundle = collection.export_bundle(b, smoke_cfg, "pc-b")
    with pytest.raises(collection.ImportConflict) as info:
        collection.import_bundle(a, smoke_cfg, bundle)
    assert info.value.conflicts
    assert sorted(p for p in a.rglob("*") if p.is_file()) == before
    kept = collection.import_bundle(a, smoke_cfg, bundle, keep_existing=True)
    assert kept["conflicts_kept_existing"] == info.value.conflicts


def forged_bundle(path, cfg, name, data=b"x", **changes):
    manifest = {
        "schema_version": records.SCHEMA_VERSION, "kind": "bundle",
        "experiment_id": cfg["experiment"]["id"], "scientific_hash": scientific_hash(cfg),
        "code_hash": collection.code_hash(), "machine": "evil", "created_utc": "2026-10-01T00:00:00+00:00",
        "files": {name: records.sha256_bytes(data)}, **changes,
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("bundle.json", json.dumps(manifest))
        z.writestr(name, data)
    return path


@pytest.mark.parametrize("name", ["runs/smoke-pilot-v0/../../evil.txt", "/etc/evil", "runs/other-exp/x.json",
                                  "src/robustness_gap/cli.py"])
def test_import_rejects_paths_outside_experiment(tmp_path, smoke_cfg, name):
    with pytest.raises(collection.LabError, match="outside"):
        collection.import_bundle(tmp_path, smoke_cfg, forged_bundle(tmp_path / "b.zip", smoke_cfg, name))


@pytest.mark.parametrize("machine", ["../../escaped", "..\\..\\escaped", "", None])
def test_import_rejects_bad_machine_name_before_writing(tmp_path, smoke_cfg, machine):
    root = tmp_path / "root"
    bundle = forged_bundle(tmp_path / "b.zip", smoke_cfg, "runs/smoke-pilot-v0/x.json", machine=machine)
    with pytest.raises(collection.LabError, match="machine name"):
        collection.import_bundle(root, smoke_cfg, bundle)
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["b.zip"]


def test_import_rejects_wrong_settings_code_or_damage(tmp_path, smoke_cfg):
    name = "runs/smoke-pilot-v0/x.json"
    with pytest.raises(collection.LabError, match="scientific"):
        collection.import_bundle(tmp_path, smoke_cfg, forged_bundle(tmp_path / "1.zip", smoke_cfg, name, scientific_hash="0"))
    with pytest.raises(collection.LabError, match="code"):
        collection.import_bundle(tmp_path, smoke_cfg, forged_bundle(tmp_path / "2.zip", smoke_cfg, name, code_hash="0"))
    damaged = forged_bundle(tmp_path / "3.zip", smoke_cfg, name, files={name: "0" * 64})
    with pytest.raises(collection.LabError, match="damaged"):
        collection.import_bundle(tmp_path, smoke_cfg, damaged)


def test_selfcheck_fingerprint_is_repeatable(tmp_path, smoke_cfg):
    first = lab.selfcheck(tmp_path, smoke_cfg, "pc-a", run_tests=False)
    second = lab.selfcheck(tmp_path, smoke_cfg, "pc-a", run_tests=False)
    assert first["fingerprint"] == second["fingerprint"]
    assert collection.progress(tmp_path, smoke_cfg)["machines"]["pc-a"]["fingerprint"] == first["fingerprint"][:12]


def test_cli_end_to_end(tmp_path, capsys):
    common = ["--config", str(REPO / "configs/pilot.yaml"), "--override",
              str(REPO / "configs/overrides/smoke.yaml"), "--root", str(tmp_path)]
    assert main(["plan", *common, "--machines", "2"]) == 0
    assert main(["work", *common, "--machine", "pc-a", "--all"]) == 0
    assert main(["note", *common, "--machine", "pc-a", "fan", "was", "loud"]) == 0
    assert main(["progress", *common]) == 0
    assert main(["analyze", *common]) == 0
    out = capsys.readouterr().out
    assert "slot  2/2" in out and "complete: True" in out
    assert main(["work", *common, "--machine", "bad name", "--all"]) == 1


def test_main_config_work_is_blocked(tmp_path):
    assert main(["work", "--config", str(REPO / "configs/main.yaml"), "--root", str(tmp_path),
                 "--machine", "pc-a", "--all", "--main-run"]) == 1

