from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import records
from .config import scientific_hash
from .provenance import code_hash

MACHINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
LATER_STAGES = ("certificate", "attack", "reference")
SUMMARY_DIR = "summary"


class LabError(RuntimeError):
    pass


class ImportConflict(LabError):
    def __init__(self, conflicts: list[str]):
        self.conflicts = conflicts
        listing = "\n  ".join(conflicts[:20])
        more = f"\n  ... and {len(conflicts) - 20} more" if len(conflicts) > 20 else ""
        super().__init__(f"{len(conflicts)} conflicting files:\n  {listing}{more}")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def check_machine(name: str) -> str:
    if not name or not MACHINE_NAME.match(name):
        raise LabError(f"machine name {name!r} must be 1-40 letters, digits, '-' or '_'")
    return name


def suggest_machine_name(hostname: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", hostname).strip("-_")[:40]
    return cleaned.lower() if cleaned else "machine"


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


def job_by_id(cfg: dict, job_id: str) -> Job:
    for job in all_jobs(cfg):
        if job.job_id == job_id:
            return job
    raise LabError(f"unknown job {job_id!r}")


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


def panel_ids(root: str | Path, cfg: dict) -> list[str]:
    path = Paths(root, cfg).results / "panel.json"
    return records.read_json(path)["sample_ids"] if path.exists() else []


def latest_selfcheck(machine_dir: Path) -> dict | None:
    checks = sorted((machine_dir / "selfcheck").glob("*.json"))
    return records.read_json(checks[-1]) if checks else None


def progress(root: str | Path, cfg: dict) -> dict:
    paths = Paths(root, cfg)
    panel = panel_ids(root, cfg)
    rows = []
    for job in all_jobs(cfg):
        rec_path = paths.model_dir(job.job_id) / "record.json"
        rec = records.read_json(rec_path) if rec_path.exists() else None
        counts = {st: sum((paths.instance_dir(job.job_id, s) / f"{st}.json").exists() for s in panel)
                  for st in ("clean", *LATER_STAGES)}
        rows.append({
            "job": job.job_id,
            "trained": rec["status"] if rec else "not started",
            "machine": rec.get("machine") if rec else None,
            "validation_accuracy": rec.get("validation_accuracy") if rec else None,
            **counts,
        })
    machines = {}
    for d in sorted(p for p in (paths.runs / "machines").glob("*") if p.is_dir()):
        latest = latest_selfcheck(d)
        machines[d.name] = {
            "sessions": len(list((d / "sessions").glob("*.jsonl"))),
            "notes": len(read_jsonl(d / "notes.jsonl")),
            "cpu": latest["environment"]["cpu_model"] if latest else None,
            "tests_passed": latest["tests"].get("passed") if latest else None,
            "fingerprint": latest["fingerprint"][:12] if latest and latest["fingerprint"] else None,
            "fingerprint_seconds": latest["fingerprint_seconds"] if latest else None,
        }
    return {"panel_size": len(panel), "jobs": rows, "machines": machines}


def instance_rows(root: str | Path, cfg: dict) -> list[dict]:
    paths = Paths(root, cfg)
    rows = []
    for job in all_jobs(cfg):
        for sid in panel_ids(root, cfg):
            folder = paths.instance_dir(job.job_id, sid)
            clean = folder / "clean.json"
            if not clean.exists():
                continue
            r = records.read_json(clean)
            rows.append({
                "job": job.job_id,
                "sample_id": sid,
                "x": r["x"],
                "true_label": r["true_label"],
                "predicted_label": r["predicted_label"],
                "clean_margin": r["clean_margin"],
                "correct": r["correct"],
                "machine": r["machine"],
                "stages": {st: (folder / f"{st}.json").exists() for st in LATER_STAGES},
                "folder": folder,
            })
    return rows


def _bundle_files(paths: Paths) -> list[Path]:
    files = []
    for base in (paths.runs, paths.results):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if p.is_file() and not p.name.startswith(".") and rel.parts[0] not in (SUMMARY_DIR, "server"):
                files.append(p)
    return files


def export_bundle(root: str | Path, cfg: dict, machine: str) -> Path:
    check_machine(machine)
    root = Path(root)
    paths = Paths(root, cfg)
    files = _bundle_files(paths)
    if not files:
        raise LabError(f"nothing to export for experiment {paths.experiment_id}")
    manifest = {
        "schema_version": records.SCHEMA_VERSION,
        "kind": "bundle",
        "experiment_id": paths.experiment_id,
        "scientific_hash": scientific_hash(cfg),
        "code_hash": code_hash(),
        "machine": machine,
        "created_utc": now_utc(),
        "files": {p.relative_to(root).as_posix(): records.sha256_file(p) for p in files},
    }
    out = paths.exports / f"{paths.experiment_id}__{machine}__{stamp()}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", records.dumps(manifest))
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    tmp.replace(out)
    return out


def _safe_member(name: str, experiment_id: str) -> PurePosixPath:
    path = PurePosixPath(name)
    allowed = (("runs", experiment_id), ("results", experiment_id))
    if path.is_absolute() or ".." in path.parts or path.parts[:2] not in allowed or len(path.parts) < 3:
        raise LabError(f"bundle contains a path outside the experiment folders: {name!r}")
    if path.parts[2] in (SUMMARY_DIR, "server"):
        raise LabError(f"bundle contains a reserved path: {name!r}")
    return path


def import_bundle(root: str | Path, cfg: dict, bundle: str | Path, keep_existing: bool = False,
                  allow_code_mismatch: bool = False) -> dict:
    root = Path(root)
    paths = Paths(root, cfg)
    try:
        z = zipfile.ZipFile(bundle)
    except zipfile.BadZipFile as e:
        raise LabError(f"not a valid zip file: {e}") from e
    with z:
        try:
            manifest = json.loads(z.read("bundle.json"))
        except KeyError as e:
            raise LabError("not a bundle: bundle.json is missing") from e
        if manifest.get("kind") != "bundle" or manifest.get("schema_version") != records.SCHEMA_VERSION:
            raise LabError("not a bundle made by this version of the tool")
        if manifest["experiment_id"] != paths.experiment_id:
            raise LabError(f"bundle is for experiment {manifest['experiment_id']!r}, "
                           f"config is {paths.experiment_id!r}")
        if manifest["scientific_hash"] != scientific_hash(cfg):
            raise LabError("bundle was produced with different scientific settings")
        if manifest["code_hash"] != code_hash() and not allow_code_mismatch:
            raise LabError("bundle was produced by different code (code hash differs)")
        listed = set(manifest["files"])
        extra = set(z.namelist()) - listed - {"bundle.json"}
        if extra:
            raise LabError(f"bundle contains unlisted files: {sorted(extra)[:5]}")

        added, identical, extended, conflicts, writes = [], [], [], [], []
        for name, digest in sorted(manifest["files"].items()):
            member = _safe_member(name, paths.experiment_id)
            data = z.read(name)
            if records.sha256_bytes(data) != digest:
                raise LabError(f"{name}: content does not match its hash; the bundle is damaged")
            target = root.joinpath(*member.parts)
            if not target.exists():
                added.append(name)
                writes.append((target, data))
                continue
            existing = target.read_bytes()
            if existing == data:
                identical.append(name)
            elif name.endswith(".jsonl") and data.startswith(existing):
                extended.append(name)
                writes.append((target, data))
            elif name.endswith(".jsonl") and existing.startswith(data):
                identical.append(name)
            else:
                conflicts.append(name)

    if conflicts and not keep_existing:
        raise ImportConflict(conflicts)
    for target, data in writes:
        records.atomic_write_bytes(target, data)
    report = {
        "time": now_utc(),
        "bundle": Path(bundle).name,
        "from_machine": manifest["machine"],
        "bundle_created_utc": manifest["created_utc"],
        "code_hash_matched": manifest["code_hash"] == code_hash(),
        "added": len(added),
        "identical": len(identical),
        "extended_logs": len(extended),
        "conflicts_kept_existing": conflicts,
    }
    records.atomic_write_json(paths.runs / "imports" / f"{stamp()}__{manifest['machine']}.json", report)
    return report
