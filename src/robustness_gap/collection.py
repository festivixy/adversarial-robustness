from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import records

MACHINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
LATER_STAGES = ("certificate", "attack", "reference")


class LabError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def check_machine(name: str) -> str:
    if not name or not MACHINE_NAME.match(name):
        raise LabError(f"machine name {name!r} must be 1-40 letters, digits, '-' or '_'")
    return name


def model_id(arch_id: str, seed: int) -> str:
    return f"{arch_id}-seed{seed}"


@dataclass(frozen=True)
class Job:
    job_id: str
    arch_id: str
    seed: int


def all_jobs(cfg: dict) -> list[Job]:
    return [
        Job(model_id(arch_id, seed), arch_id, seed)
        for arch_id in cfg["model"]["architectures"]
        for seed in cfg["training"]["seeds"]
    ]


def parse_slot(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not match:
        raise LabError(f"slot {text!r} must look like 3/10")
    k, n = int(match.group(1)), int(match.group(2))
    if not 1 <= k <= n:
        raise LabError(f"slot {k}/{n}: the first number must be between 1 and {n}")
    return k, n


def assign(jobs: list[Job], n_machines: int) -> dict[int, list[Job]]:
    if n_machines < 1:
        raise LabError("need at least one machine")
    return {k: [j for i, j in enumerate(jobs) if i % n_machines == k - 1] for k in range(1, n_machines + 1)}


def select_jobs(cfg: dict, job_ids: list[str] | None = None, slot: str | None = None,
                everything: bool = False) -> list[Job]:
    modes = sum([bool(job_ids), slot is not None, everything])
    if modes != 1:
        raise LabError("choose exactly one of: specific jobs, a slot, or all jobs")
    jobs = all_jobs(cfg)
    if everything:
        return jobs
    if slot is not None:
        k, n = parse_slot(slot)
        return assign(jobs, n)[k]
    by_id = {j.job_id: j for j in jobs}
    unknown = [j for j in job_ids if j not in by_id]
    if unknown:
        raise LabError(f"unknown jobs {unknown}; known jobs are {list(by_id)}")
    return [by_id[j] for j in job_ids]


class Paths:
    def __init__(self, root: str | Path, cfg: dict):
        self.root = Path(root)
        self.experiment_id = cfg["experiment"]["id"]
        self.runs = self.root / "runs" / self.experiment_id
        self.results = self.root / "results" / self.experiment_id
        self.data_root = self.root / "data"
        self.exports = self.root / "exports"

    def model_dir(self, job_id: str) -> Path:
        return self.runs / "models" / job_id

    def instance_dir(self, job_id: str, sample_id: str) -> Path:
        return self.results / "instances" / job_id / sample_id

    def machine_dir(self, machine: str) -> Path:
        return self.runs / "machines" / check_machine(machine)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = records.dumps(obj, canonical=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EventLog:
    def __init__(self, path: Path, machine: str):
        self.path = path
        self.machine = machine

    def write(self, event: str, **fields) -> None:
        append_jsonl(self.path, {"time": now_utc(), "machine": self.machine, "event": event, **fields})


def add_note(root: str | Path, cfg: dict, machine: str, text: str) -> None:
    if not text.strip():
        raise LabError("empty note")
    append_jsonl(Paths(root, cfg).machine_dir(machine) / "notes.jsonl",
                 {"time": now_utc(), "machine": machine, "note": text.strip()})


def all_notes(root: str | Path, cfg: dict) -> list[dict]:
    machines = Paths(root, cfg).runs / "machines"
    notes = [n for p in sorted(machines.glob("*/notes.jsonl")) for n in read_jsonl(p)]
    return sorted(notes, key=lambda n: (n["time"], n["machine"]))
