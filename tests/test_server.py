import io
import json
import threading
import urllib.request
import zipfile

import pytest

from robustness_gap import collection, records
from robustness_gap.config import load_config
from robustness_gap.provenance import code_hash
from robustness_gap.server import Coordinator, CoordinatorError, make_server
from robustness_gap.worker import CoordinatorClient, RemoteError, serve_jobs

from conftest import REPO


@pytest.fixture
def smoke_cfg():
    return load_config(REPO / "configs/pilot.yaml", [REPO / "configs/overrides/smoke.yaml"])[1]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def ident(cfg, machine="pc-a", **changes):
    from robustness_gap.config import scientific_hash

    return {"machine": machine, "scientific_hash": scientific_hash(cfg), "code_hash": code_hash(), **changes}


def test_claims_follow_job_order_and_run_out(tmp_path, smoke_cfg):
    c = Coordinator(tmp_path, smoke_cfg, token="abc")
    got = [c.claim(ident(smoke_cfg))["job"] for _ in range(3)]
    assert got == [j.job_id for j in collection.all_jobs(smoke_cfg)]
    assert c.claim(ident(smoke_cfg)) == {"job": None, "unfinished": 3}


def test_expired_lease_returns_job_to_queue(tmp_path, smoke_cfg):
    clock = Clock()
    c = Coordinator(tmp_path, smoke_cfg, token="abc", lease_seconds=60, clock=clock)
    job = c.claim(ident(smoke_cfg, "pc-a"))["job"]
    clock.t += 30
    assert c.heartbeat({**ident(smoke_cfg, "pc-a"), "job": job})["ok"]
    clock.t += 61
    assert c.claim(ident(smoke_cfg, "pc-b"))["job"] == job
    assert not c.heartbeat({**ident(smoke_cfg, "pc-a"), "job": job})["ok"]
    history = [h["event"] for h in c.state["jobs"][job]["history"]]
    assert history == ["claimed", "lease_expired", "claimed"]


def test_mismatched_clients_are_turned_away(tmp_path, smoke_cfg):
    c = Coordinator(tmp_path, smoke_cfg, token="abc")
    with pytest.raises(CoordinatorError, match="different code"):
        c.claim(ident(smoke_cfg, code_hash="0"))
    with pytest.raises(CoordinatorError, match="different experiment settings"):
        c.claim(ident(smoke_cfg, scientific_hash="0"))
    with pytest.raises(collection.LabError):
        c.claim(ident(smoke_cfg, machine="bad name"))


def test_done_requires_uploaded_results_and_failures_stay_failed(tmp_path, smoke_cfg):
    c = Coordinator(tmp_path, smoke_cfg, token="abc")
    job = c.claim(ident(smoke_cfg))["job"]
    with pytest.raises(CoordinatorError, match="uploaded"):
        c.report({**ident(smoke_cfg), "job": job, "outcome": "done"})
    c.report({**ident(smoke_cfg), "job": job, "outcome": "failed", "error": "boom"})
    claimed = [c.claim(ident(smoke_cfg))["job"] for _ in range(3)]
    assert job not in claimed
    restarted = Coordinator(tmp_path, smoke_cfg, token="abc", reset_failed=True)
    assert restarted.claim(ident(smoke_cfg))["job"] == job


def test_queue_survives_restart_and_rejects_changed_settings(tmp_path, smoke_cfg):
    c = Coordinator(tmp_path, smoke_cfg)
    job = c.claim(ident(smoke_cfg))["job"]
    again = Coordinator(tmp_path, smoke_cfg)
    assert again.token == c.token and again.state["jobs"][job]["state"] == "claimed"
    changed = json.loads(json.dumps(smoke_cfg))
    changed["training"]["learning_rate"] = 0.5
    with pytest.raises(CoordinatorError, match="different settings"):
        Coordinator(tmp_path, changed)


@pytest.fixture
def running(tmp_path, smoke_cfg):
    coordinator = Coordinator(tmp_path / "pi", smoke_cfg, token="abc")
    server = make_server(coordinator, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield coordinator, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_http_requires_access_code(running, smoke_cfg):
    _, url = running
    with pytest.raises(RemoteError, match="access code"):
        CoordinatorClient(url, "wrong").claim(ident(smoke_cfg))
    assert CoordinatorClient(url, "abc").status()["counts"]["pending"] == 3


def test_status_page_and_project_download(running):
    _, url = running
    page = urllib.request.urlopen(f"{url}/").read().decode()
    assert "smoke-pilot-v0" in page and "3 waiting" in page
    script = urllib.request.urlopen(f"{url}/bootstrap.ps1").read().decode()
    assert f"{url}/project.zip" in script


def test_project_zip_contains_code(tmp_path, smoke_cfg):
    c = Coordinator(tmp_path, smoke_cfg, token="abc")
    names = zipfile.ZipFile(io.BytesIO(c.project_zip())).namelist()
    assert "src/robustness_gap/server.py" in names and "uv.lock" in names
    assert not any("__pycache__" in n or n.startswith(("runs/", "results/", "data/")) for n in names)


def test_two_workers_finish_the_queue(running, tmp_path, smoke_cfg):
    coordinator, url = running
    client = CoordinatorClient(url, "abc")
    results = {}

    def go(name):
        results[name] = serve_jobs(tmp_path / name, smoke_cfg, client, name, say=lambda *_: None)

    threads = [threading.Thread(target=go, args=(n,)) for n in ("pc-a", "pc-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    status = client.status()
    assert status["counts"]["done"] == 3
    ran = sorted(s["job"] for r in results.values() for s in r)
    assert ran == sorted(j.job_id for j in collection.all_jobs(smoke_cfg))
    report = collection.progress(tmp_path / "pi", smoke_cfg)
    assert all(r["trained"] == "success" and r["clean"] == smoke_cfg["panel"]["size"] for r in report["jobs"])
    uploads = list((tmp_path / "pi/runs/smoke-pilot-v0/server/uploads").glob("*.zip"))
    assert len(uploads) == 3


def test_worker_refuses_mismatched_coordinator(running, tmp_path, smoke_cfg):
    _, url = running
    other = json.loads(json.dumps(smoke_cfg))
    other["training"]["learning_rate"] = 0.5
    with pytest.raises(RemoteError, match="different settings"):
        serve_jobs(tmp_path / "x", other, CoordinatorClient(url, "abc"), "pc-a", say=lambda *_: None)


def test_unreachable_coordinator_is_a_clear_error(tmp_path, smoke_cfg):
    with pytest.raises(RemoteError, match="cannot reach"):
        CoordinatorClient("http://127.0.0.1:9", "abc", timeout=2).status()
