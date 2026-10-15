from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import records
from .collection import EventLog, Job, LATER_STAGES, Paths, check_machine, now_utc, stamp
from .config import scientific_hash
from .data import Dataset, generate_dataset, load_dataset, select_panel
from .provenance import code_hash, code_identity, cpu_model, environment
from .targets import TargetDefinition, predict
from .train import DTYPES, fit, load_network, state_fingerprint, train_one

SELFCHECK_EPOCHS = 5


def prepare(root: str | Path, cfg: dict) -> tuple[Dataset, str, list[str]]:
    paths = Paths(root, cfg)
    ds_path = generate_dataset(cfg, paths.data_root)
    ds = load_dataset(ds_path)
    p = cfg["panel"]
    panel = select_panel(ds, p["size"], p["source_split"], p["selection_seed"], p["class_balanced"])
    records.write_new_json(paths.runs / "config.resolved.json", {
        "scientific_hash": scientific_hash(cfg), "config": cfg,
    })
    records.write_new_json(paths.results / "panel.json", records.make_record("panel", {
        "experiment_id": paths.experiment_id, "dataset_id": ds.dataset_id, "sample_ids": panel, **p,
    }, input_hash=records.sha256_json({"dataset_id": ds.dataset_id, "panel": p})))
    return ds, records.sha256_file(ds_path / "arrays.npz"), panel


def clean_stage(paths: Paths, cfg: dict, ds: Dataset, index: dict, rec: dict, net, sample_id: str,
                machine: str) -> bool:
    path = paths.instance_dir(rec["model_id"], sample_id) / "clean.json"
    input_hash = records.sha256_json({
        "stage": "clean", "checkpoint_hash": rec["checkpoint_hash"],
        "dataset_hash": rec["dataset_hash"], "sample_id": sample_id,
    })
    if records.load_if_current(path, input_hash) is not None:
        return False
    i = index[sample_id]
    x, y = ds.X[i], int(ds.y[i])
    logits = net.logits(x)
    pred = predict(logits)
    records.atomic_write_json(path, records.make_record("instance_clean", {
        "experiment_id": paths.experiment_id,
        "model_id": rec["model_id"],
        "sample_id": sample_id,
        "x": x,
        "true_label": y,
        "predicted_label": pred,
        "logits": logits,
        "clean_margin": float(logits[pred] - logits[1 - pred]),
        "correct": pred == y,
        "domain_id": cfg["domain"]["id"],
        "machine": machine,
    }, input_hash))
    return True


def run_job(root: str, cfg: dict, job: Job, machine: str, provenance: dict) -> dict:
    t0 = time.perf_counter()
    paths = Paths(root, cfg)
    ds, ds_hash, panel = prepare(root, cfg)
    rec = train_one(cfg, ds, ds_hash, job.arch_id, job.seed, paths.model_dir(job.job_id), provenance)
    summary = {
        "job": job.job_id,
        "train_status": rec["status"],
        "trained_on": rec.get("machine"),
        "selected_epoch": rec.get("selected_epoch"),
        "validation_accuracy": rec.get("validation_accuracy"),
        "stages": {},
    }
    if rec["status"] == "success":
        _, net = load_network(rec, paths.model_dir(job.job_id))
        index = {sid: i for i, sid in enumerate(ds.ids)}
        new = sum(clean_stage(paths, cfg, ds, index, rec, net, sid, machine) for sid in panel)
        summary["stages"]["clean"] = {"new": new, "already_done": len(panel) - new}
        target = TargetDefinition.from_config(cfg["target"])
        reason = "not implemented yet" if target.resolved else "blocked until target.policy is decided (D001)"
        for stage in LATER_STAGES:
            summary["stages"][stage] = reason
    summary["seconds"] = time.perf_counter() - t0
    return summary


def machine_provenance(root: str | Path, machine: str) -> dict:
    return {**code_identity(Path(root)), "code_hash": code_hash(), "machine": machine, "cpu_model": cpu_model()}


def describe(summary: dict) -> str:
    if "error" in summary:
        return f"FAILED  {summary['job']:<16} {summary['error']}"
    acc = summary["validation_accuracy"]
    acc_text = f"{acc:.3f}" if isinstance(acc, float) else "-"
    return (f"done    {summary['job']:<16} train={summary['train_status']} val_acc={acc_text} "
            f"({summary['seconds']:.1f}s)")


def work(root: str | Path, cfg: dict, jobs: list[Job], machine: str, say=print) -> list[dict]:
    check_machine(machine)
    root = Path(root)
    paths = Paths(root, cfg)
    prepare(root, cfg)
    provenance = machine_provenance(root, machine)
    events = EventLog(paths.machine_dir(machine) / "sessions" / f"{stamp()}.jsonl", machine)
    events.write("session_start", jobs=[j.job_id for j in jobs], workers=1,
                 config_hash=scientific_hash(cfg), code_hash=provenance["code_hash"],
                 code_revision=provenance["code_revision"], cpu_model=provenance["cpu_model"],
                 logical_cpus=os.cpu_count())
    summaries: list[dict] = []
    try:
        for job in jobs:
            events.write("job_start", job=job.job_id)
            say(f"start   {job.job_id}")
            try:
                summary = run_job(str(root), cfg, job, machine, provenance)
                events.write("job_done", job=job.job_id, summary=summary)
            except Exception as e:
                summary = {"job": job.job_id, "error": f"{type(e).__name__}: {e}"}
                events.write("job_failed", job=job.job_id, error=summary["error"])
            summaries.append(summary)
            say(describe(summary))
    except KeyboardInterrupt:
        events.write("session_interrupted", completed=[s["job"] for s in summaries])
        raise
    events.write("session_end", completed=len([s for s in summaries if "error" not in s]),
                 failed=len([s for s in summaries if "error" in s]))
    return summaries


def selfcheck(root: str | Path, cfg: dict, machine: str, run_tests: bool = True) -> dict:
    check_machine(machine)
    root = Path(root)
    tests = {"ran": False}
    if run_tests and (root / "tests").is_dir():
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(root / "tests")],
                           capture_output=True, text=True, cwd=root)
        lines = [line for line in r.stdout.splitlines() if line.strip()]
        tests = {"ran": True, "passed": r.returncode == 0, "summary": lines[-1] if lines else ""}

    ds = load_dataset(generate_dataset(cfg, Paths(root, cfg).data_root))
    arch_id = next(iter(cfg["model"]["architectures"]))
    widths = cfg["model"]["architectures"][arch_id]
    result = fit(cfg["training"], widths, DTYPES[cfg["model"]["train_dtype"]], ds, 0, SELFCHECK_EPOCHS)
    report = {
        "machine": machine,
        "time": now_utc(),
        "environment": environment(root),
        "tests": tests,
        "fingerprint_network": f"{arch_id}, seed 0, {SELFCHECK_EPOCHS} epochs",
        "fingerprint": state_fingerprint(result.best_state) if result.best_state else None,
        "fingerprint_seconds": result.elapsed,
    }
    records.atomic_write_json(Paths(root, cfg).machine_dir(machine) / "selfcheck" / f"{stamp()}.json", report)
    return report
