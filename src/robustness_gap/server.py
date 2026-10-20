from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import records
from .collection import EventLog, LabError, Paths, all_jobs, check_machine
from .config import scientific_hash
from .provenance import code_hash

DEFAULT_PORT = 8765
DEFAULT_LEASE_SECONDS = 600


class CoordinatorError(LabError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds") if t is not None else None


class Coordinator:
    def __init__(self, root: str | Path, cfg: dict, token: str | None = None,
                 lease_seconds: float = DEFAULT_LEASE_SECONDS, reset_failed: bool = False, clock=time.time):
        self.root = Path(root)
        self.cfg = cfg
        self.paths = Paths(root, cfg)
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.lock = threading.Lock()
        self.scientific_hash = scientific_hash(cfg)
        self.code_hash = code_hash()
        self.dir = self.paths.runs / "server"
        self.events = EventLog(self.dir / "events.jsonl", "server")
        self.token = token or self._stored_token()
        self.state = self._load_state()
        self.events.write("coordinator_start", lease_seconds=lease_seconds, code_hash=self.code_hash,
                          scientific_hash=self.scientific_hash)
        if reset_failed:
            for job_id, job in self.state["jobs"].items():
                if job["state"] == "failed":
                    job.update(state="pending", machine=None, lease_until=None)
                    job["history"].append({"time": iso(self.clock()), "event": "reset_after_failure"})
                    self.events.write("job_reset", job=job_id)
            self._save()

    def _stored_token(self) -> str:
        path = self.dir / "access-code.txt"
        if path.exists():
            return path.read_text().strip()
        code = secrets.token_hex(3)
        records.atomic_write_bytes(path, (code + "\n").encode())
        return code

    def _load_state(self) -> dict:
        path = self.dir / "queue.json"
        ids = [j.job_id for j in all_jobs(self.cfg)]
        if path.exists():
            state = records.read_json(path)
            if list(state["jobs"]) != ids or state["scientific_hash"] != self.scientific_hash:
                raise CoordinatorError(f"{path} belongs to different settings; use a new experiment id")
            return state
        state = {
            "experiment_id": self.paths.experiment_id,
            "scientific_hash": self.scientific_hash,
            "jobs": {i: {"state": "pending", "machine": None, "lease_until": None, "attempts": 0,
                         "error": None, "history": []} for i in ids},
            "machines": {},
        }
        records.atomic_write_json(path, state)
        return state

    def _save(self) -> None:
        records.atomic_write_json(self.dir / "queue.json", self.state)

    def _job(self, job_id: str) -> dict:
        if job_id not in self.state["jobs"]:
            raise CoordinatorError(f"unknown job {job_id!r}", 404)
        return self.state["jobs"][job_id]

    def _seen(self, machine: str, **info) -> None:
        entry = self.state["machines"].setdefault(machine, {"first_seen": iso(self.clock()), "jobs": []})
        entry["last_seen"] = iso(self.clock())
        entry.update(info)

    def _expire(self) -> None:
        now = self.clock()
        for job_id, job in self.state["jobs"].items():
            if job["state"] == "claimed" and job["lease_until"] < now:
                job["history"].append({"time": iso(now), "event": "lease_expired", "machine": job["machine"]})
                self.events.write("lease_expired", job=job_id, from_machine=job["machine"])
                job.update(state="pending", machine=None, lease_until=None)

    def check_auth(self, token: str | None) -> None:
        if not token or not hmac.compare_digest(token, self.token):
            raise CoordinatorError("wrong or missing access code", 401)

    def check_client(self, body: dict) -> str:
        machine = check_machine(body.get("machine", ""))
        if body.get("scientific_hash") != self.scientific_hash:
            raise CoordinatorError("this computer has different experiment settings than the coordinator", 409)
        if body.get("code_hash") != self.code_hash:
            raise CoordinatorError("this computer runs different code than the coordinator; "
                                   "reinstall it from the coordinator", 409)
        return machine

    def claim(self, body: dict) -> dict:
        machine = self.check_client(body)
        with self.lock:
            self._expire()
            self._seen(machine, cpu=body.get("cpu"))
            for job_id, job in self.state["jobs"].items():
                if job["state"] == "pending":
                    job.update(state="claimed", machine=machine, lease_until=self.clock() + self.lease_seconds)
                    job["attempts"] += 1
                    job["history"].append({"time": iso(self.clock()), "event": "claimed", "machine": machine})
                    self.state["machines"][machine]["jobs"].append(job_id)
                    self.events.write("job_claimed", job=job_id, by=machine, attempt=job["attempts"])
                    self._save()
                    return {"job": job_id, "lease_seconds": self.lease_seconds}
            self._save()
            remaining = sum(j["state"] in ("pending", "claimed") for j in self.state["jobs"].values())
            return {"job": None, "unfinished": remaining}

    def heartbeat(self, body: dict) -> dict:
        machine = self.check_client(body)
        with self.lock:
            self._expire()
            self._seen(machine)
            job = self._job(body.get("job", ""))
            if job["state"] != "claimed" or job["machine"] != machine:
                self._save()
                return {"ok": False, "reason": "this computer no longer holds the lease for the job"}
            job["lease_until"] = self.clock() + self.lease_seconds
            self._save()
            return {"ok": True}

    def report(self, body: dict) -> dict:
        machine = self.check_client(body)
        outcome = body.get("outcome")
        if outcome not in ("done", "failed"):
            raise CoordinatorError("outcome must be 'done' or 'failed'")
        with self.lock:
            self._seen(machine)
            job_id = body.get("job", "")
            job = self._job(job_id)
            record = self.paths.model_dir(job_id) / "record.json"
            if outcome == "done" and not record.exists():
                raise CoordinatorError(f"no results for {job_id} have been uploaded yet", 409)
            if job["state"] == "done":
                job["history"].append({"time": iso(self.clock()), "event": f"late_{outcome}", "machine": machine})
                self.events.write("late_report", job=job_id, by=machine, outcome=outcome)
                self._save()
                return {"ok": True, "note": "job was already done; the earlier result is kept"}
            job.update(state=outcome, machine=machine, lease_until=None, error=body.get("error"))
            job["history"].append({"time": iso(self.clock()), "event": outcome, "machine": machine})
            self.events.write(f"job_{outcome}", job=job_id, by=machine, error=body.get("error"))
            self._save()
            return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    coordinator: Coordinator
    server_version = "robustness-gap-coordinator"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, records.dumps(obj).encode(), "application/json")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length)

    def do_POST(self):
        c = self.coordinator
        try:
            c.check_auth(self.headers.get("X-Access-Code"))
            body = json.loads(self._body() or b"{}")
            actions = {"/api/claim": c.claim, "/api/heartbeat": c.heartbeat, "/api/report": c.report}
            if self.path not in actions:
                self._json(404, {"error": "not found"})
                return
            self._json(200, actions[self.path](body))
        except CoordinatorError as e:
            self._json(e.status, {"error": str(e)})
        except (LabError, ValueError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(coordinator: Coordinator, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"coordinator": coordinator})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
