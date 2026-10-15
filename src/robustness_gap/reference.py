from __future__ import annotations

from dataclasses import dataclass

from .records import Status


@dataclass
class ReferenceResult:
    raw_lower_bound: float | None
    raw_incumbent_objective: float | None
    validated_upper_bound: float | None
    candidate: list[float] | None
    termination_status: Status
    solver_status_raw: str | None
    mip_gap: float | None
    solver_message: str | None
    elapsed_seconds: float | None
    time_limit: float
    resolved: bool
    interval_width: float | None
    validation_status: str | None


def reference(*args, **kwargs) -> ReferenceResult:
    raise NotImplementedError("reference adapter is blocked on decisions D001 and D005")
