from __future__ import annotations

from dataclasses import dataclass, field

from .records import Status


@dataclass
class CertificateResult:
    L: float | None
    certificate_status: Status
    solver_status: str | None
    radius_search_ceiling: float | None
    ceiling_reached: bool
    last_certified_radius: float | None
    search_trace: list[dict] = field(default_factory=list)
    bound_method: str = "interval+triangle_lp"
    tolerance_id: str | None = None
    elapsed_seconds: float | None = None


def certify(*args, **kwargs) -> CertificateResult:
    raise NotImplementedError("certificate adapter is blocked on decisions D001 and D005")
