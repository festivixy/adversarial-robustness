from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .bounds import DOMAIN_HI, DOMAIN_LO
from .models import INPUT_DIM, AffineReLUNetwork
from .targets import TargetDefinition, evaluate_target


@dataclass(frozen=True)
class CandidateValidation:
    finite: bool
    correct_dimension: bool
    valid_domain: bool
    recomputed_distance: float | None
    logits: tuple[float, float] | None
    predicted_class: int | None
    score_condition_satisfied: bool | None
    prediction_changed: bool | None
    valid_target: bool
    valid: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def linf_distance(x: np.ndarray, x_prime: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x_prime, np.float64) - np.asarray(x, np.float64))))


def _rejected(reason: str, **fields) -> CandidateValidation:
    base = dict(
        finite=False, correct_dimension=False, valid_domain=False, recomputed_distance=None,
        logits=None, predicted_class=None, score_condition_satisfied=None,
        prediction_changed=None, valid_target=False, valid=False, reason=reason,
    )
    base.update(fields)
    return CandidateValidation(**base)


def validate_candidate(
    net: AffineReLUNetwork,
    x: np.ndarray,
    candidate: np.ndarray,
    original_class: int,
    target: TargetDefinition,
    domain_tol: float,
) -> CandidateValidation:
    target.require_resolved()
    cand = np.asarray(candidate, dtype=np.float64)
    if cand.shape != (INPUT_DIM,):
        return _rejected(f"bad shape {cand.shape}", finite=bool(np.all(np.isfinite(cand))))
    if not np.all(np.isfinite(cand)):
        return _rejected("non-finite coordinates", correct_dimension=True)

    in_domain = bool(np.all(cand >= DOMAIN_LO - domain_tol) and np.all(cand <= DOMAIN_HI + domain_tol))
    ev = evaluate_target(target, net.logits(cand), original_class)
    valid = in_domain and ev.target_satisfied
    if valid:
        reason = "ok"
    elif not in_domain:
        reason = "outside domain"
    else:
        reason = "target predicate not satisfied"
    return CandidateValidation(
        finite=True,
        correct_dimension=True,
        valid_domain=in_domain,
        recomputed_distance=linf_distance(x, cand),
        logits=ev.logits,
        predicted_class=ev.predicted_class,
        score_condition_satisfied=ev.score_condition_satisfied,
        prediction_changed=ev.prediction_changed,
        valid_target=ev.target_satisfied,
        valid=valid,
        reason=reason,
    )
