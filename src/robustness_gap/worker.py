from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from .collection import EventLog, LabError, Paths, check_machine, export_bundle, job_by_id, stamp
from .config import scientific_hash
from .lab import describe, keep_awake, machine_provenance, prepare, run_job

HEARTBEAT_SECONDS = 60
UPLOAD_ATTEMPTS = 4


class RemoteError(LabError):
    pass


class CoordinatorClient:
    def __init__(self, url: str, access_code: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            self.url = f"http://{self.url}"
        self.access_code = access_code
        self.timeout = timeout

    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", data=data, headers={
            "X-Access-Code": self.access_code, **(headers or {}),
        }, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", str(e))
            except (ValueError, AttributeError):
                message = str(e)
            raise RemoteError(f"coordinator said: {message}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RemoteError(f"cannot reach the coordinator at {self.url}: {e}") from e

    def _post(self, path: str, body: dict) -> dict:
        return self._request(path, json.dumps(body).encode(), {"Content-Type": "application/json"})

    def status(self) -> dict:
        return self._request("/api/status")

    def claim(self, ident: dict) -> dict:
        return self._post("/api/claim", ident)

    def heartbeat(self, ident: dict, job_id: str) -> dict:
        return self._post("/api/heartbeat", {**ident, "job": job_id})

    def report(self, ident: dict, job_id: str, outcome: str, error: str | None = None) -> dict:
        return self._post("/api/report", {**ident, "job": job_id, "outcome": outcome, "error": error})

    def upload(self, machine: str, bundle: Path) -> dict:
        return self._request("/api/upload", bundle.read_bytes(),
                             {"Content-Type": "application/zip", "X-Machine": machine})


def identity(cfg: dict, machine: str, provenance: dict) -> dict:
    return {"machine": machine, "scientific_hash": scientific_hash(cfg),
            "code_hash": provenance["code_hash"], "cpu": provenance["cpu_model"]}


def check_compatible(client: CoordinatorClient, cfg: dict, provenance: dict) -> dict:
    status = client.status()
    if status["experiment_id"] != cfg["experiment"]["id"]:
        raise RemoteError(f"the coordinator runs experiment {status['experiment_id']!r}, "
                          f"this computer is set to {cfg['experiment']['id']!r}")
    if status["scientific_hash"] != scientific_hash(cfg):
        raise RemoteError("the coordinator uses different settings for this experiment")
    if status["code_hash"] != provenance["code_hash"]:
        raise RemoteError("this computer runs different code than the coordinator; reinstall it from the coordinator")
    return status


def upload_with_retry(client: CoordinatorClient, machine: str, bundle: Path, events: EventLog,
                      say, pause: float = 5.0) -> bool:
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            client.upload(machine, bundle)
            return True
        except RemoteError as e:
            events.write("upload_failed", bundle=bundle.name, attempt=attempt, error=str(e))
            say(f"upload failed (attempt {attempt}/{UPLOAD_ATTEMPTS}): {e}")
            if attempt < UPLOAD_ATTEMPTS:
                time.sleep(pause * attempt)
    say(f"results kept locally in {bundle}; import it by hand later")
    return False


def serve_jobs(root: str | Path, cfg: dict, client: CoordinatorClient, machine: str, workers: int = 1,
               say=print, should_stop=lambda: False, wait_for_more: bool = False, poll_seconds: float = 30.0,
               heartbeat_seconds: float = HEARTBEAT_SECONDS, upload_pause: float = 5.0) -> list[dict]:
    check_machine(machine)
    if workers < 1:
        raise LabError("workers must be at least 1")
    root = Path(root)
    provenance = machine_provenance(root, machine)
    check_compatible(client, cfg, provenance)
    ident = identity(cfg, machine, provenance)
    prepare(root, cfg)
    events = EventLog(Paths(root, cfg).machine_dir(machine) / "sessions" / f"{stamp()}.jsonl", machine)
    events.write("session_start", mode="coordinator", coordinator=client.url, workers=workers,
                 config_hash=ident["scientific_hash"], code_hash=ident["code_hash"],
                 code_revision=provenance["code_revision"], cpu_model=provenance["cpu_model"],
                 logical_cpus=os.cpu_count())

    running: dict = {}
    beat_lock = threading.Lock()
    beating = threading.Event()

    def heartbeats() -> None:
        while not beating.wait(heartbeat_seconds):
            with beat_lock:
                job_ids = list(running.values())
            for job_id in job_ids:
                try:
                    if not client.heartbeat(ident, job_id).get("ok"):
                        events.write("lease_lost", job=job_id)
                        say(f"lease lost for {job_id}; another computer may run it too")
                except RemoteError as e:
                    events.write("heartbeat_failed", job=job_id, error=str(e))

    beater = threading.Thread(target=heartbeats, daemon=True)
    beater.start()
    summaries: list[dict] = []
    queue_empty = False

    def finish(job_id: str, summary: dict | None, error: BaseException | None) -> None:
        if error is not None:
            summary = {"job": job_id, "error": f"{type(error).__name__}: {error}"}
            events.write("job_failed", job=job_id, error=summary["error"])
        else:
            events.write("job_done", job=job_id, summary=summary)
        summaries.append(summary)
        say(describe(summary))
        bundle = export_bundle(root, cfg, machine)
        if upload_with_retry(client, machine, bundle, events, say, upload_pause):
            outcome = "failed" if error is not None else "done"
            try:
                client.report(ident, job_id, outcome, summary.get("error"))
                events.write("job_reported", job=job_id, outcome=outcome)
            except RemoteError as e:
                events.write("report_failed", job=job_id, error=str(e))
                say(f"could not report {job_id}: {e}")

    try:
        with keep_awake(), ProcessPoolExecutor(max_workers=workers,
                                               mp_context=multiprocessing.get_context("spawn")) as pool:
            while True:
                while len(running) < workers and not should_stop() and not queue_empty:
                    answer = client.claim(ident)
                    if answer["job"] is None:
                        queue_empty = True
                        break
                    job = job_by_id(cfg, answer["job"])
                    future = pool.submit(run_job, str(root), cfg, job, machine, provenance)
                    with beat_lock:
                        running[future] = job.job_id
                    events.write("job_start", job=job.job_id)
                    say(f"start   {job.job_id}")
                if not running:
                    if should_stop() or not wait_for_more:
                        break
                    say(f"no jobs available; asking again in {poll_seconds:.0f}s")
                    time.sleep(poll_seconds)
                    queue_empty = False
                    continue
                done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    with beat_lock:
                        job_id = running.pop(future)
                    try:
                        finish(job_id, future.result(), None)
                    except Exception as e:
                        finish(job_id, None, e)
    except KeyboardInterrupt:
        events.write("session_interrupted", completed=[s["job"] for s in summaries])
        raise
    finally:
        beating.set()
    if should_stop():
        events.write("session_stopped")
    events.write("session_end", completed=len([s for s in summaries if "error" not in s]),
                 failed=len([s for s in summaries if "error" in s]))
    return summaries
