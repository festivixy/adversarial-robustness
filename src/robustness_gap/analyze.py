from __future__ import annotations

from dataclasses import asdict, dataclass

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

