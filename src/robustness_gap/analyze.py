from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import records
from .collection import SUMMARY_DIR, Paths, all_jobs
from .records import Status


@dataclass(frozen=True)
class ErrorIntervals:
    A: float | None
    B: float | None
    total_gap: float | None
    certificate_error_lower: float | None
    certificate_error_upper: float | None
    attack_error_lower: float | None
    attack_error_upper: float | None
    consistency_status: Status
    note: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["consistency_status"] = self.consistency_status.value
        return d


def error_intervals(L: float | None, U: float | None, a: float | None, b: float | None,
                    tol: float) -> ErrorIntervals:
    lowers = [v for v in (a, L) if v is not None]
    uppers = [v for v in (b, U) if v is not None]
    A = max(lowers) if lowers else None
    B = min(uppers) if uppers else None
    total_gap = U - L if (U is not None and L is not None) else None

    if A is not None and B is not None and A > B + tol:
        return ErrorIntervals(A, B, total_gap, None, None, None, None, Status.INCONSISTENT_BOUNDS,
                              f"A={A!r} exceeds B={B!r} beyond tolerance {tol!r}")

    ce_lo = A - L if (L is not None and A is not None) else None
    ce_hi = B - L if (L is not None and B is not None) else None
    ae_lo = U - B if (U is not None and B is not None) else None
    ae_hi = U - A if (U is not None and A is not None) else None
    missing = [n for n, v in (("L", L), ("U", U), ("a", a), ("b", b)) if v is None]
    computed = any(v is not None for v in (ce_lo, ce_hi, ae_lo, ae_hi))
    return ErrorIntervals(
        A, B, total_gap, ce_lo, ce_hi, ae_lo, ae_hi,
        Status.SUCCESS if computed else Status.UNAVAILABLE,
        "complete" if not missing else f"missing: {', '.join(missing)}",
    )


def summarize_clean(root: str | Path, cfg: dict) -> dict:
    paths = Paths(root, cfg)
    panel_path = paths.results / "panel.json"
    if not panel_path.exists():
        raise FileNotFoundError("no panel yet; run at least one job first")
    panel = records.read_json(panel_path)["sample_ids"]
    rows = ["model_id,sample_id,true_label,predicted_label,logit0,logit1,clean_margin,correct,machine"]
    correct_by: dict[str, list[bool]] = {sid: [] for sid in panel}
    missing_models, missing_instances, failed_models = [], [], []
    for job in all_jobs(cfg):
        rec_path = paths.model_dir(job.job_id) / "record.json"
        if not rec_path.exists():
            missing_models.append(job.job_id)
            continue
        if records.read_json(rec_path)["status"] != "success":
            failed_models.append(job.job_id)
            continue
        for sid in panel:
            path = paths.instance_dir(job.job_id, sid) / "clean.json"
            if not path.exists():
                missing_instances.append(f"{job.job_id}/{sid}")
                continue
            r = records.read_json(path)
            correct_by[sid].append(r["correct"])
            rows.append(f"{r['model_id']},{sid},{r['true_label']},{r['predicted_label']},"
                        f"{r['logits'][0]!r},{r['logits'][1]!r},{r['clean_margin']!r},{r['correct']},{r['machine']}")
    complete = not (missing_models or missing_instances)
    common = sorted(sid for sid, v in correct_by.items() if v and all(v))
    summary = {
        "complete": complete,
        "panel_size": len(panel),
        "inclusion_rule": cfg["panel"]["inclusion_rule"],
        "common_correct_subset_size": len(common) if complete else None,
        "common_correct_subset": common if complete else None,
        "failed_models": failed_models,
        "missing_models": missing_models,
        "missing_instances": missing_instances,
    }
    out = paths.results / SUMMARY_DIR
    records.atomic_write_bytes(out / "clean_predictions.csv", ("\n".join(rows) + "\n").encode())
    records.atomic_write_json(out / "clean_summary.json", summary)
    return summary
