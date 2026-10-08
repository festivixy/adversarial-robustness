from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

PREDICTION_TIE_RULE = "lowest_index"


class TargetPolicy(str, Enum):
    SCORE_BOUNDARY = "score_boundary"
    POSITIVE_MARGIN = "positive_margin"
    STRICT_PREDICTION_CHANGE = "strict_prediction_change"


class UnresolvedTargetError(RuntimeError):
    pass


def predict(logits: np.ndarray) -> int:
    logits = np.asarray(logits)
    if logits.ndim != 1 or logits.shape[0] != 2:
        raise ValueError(f"expected 2 logits, got shape {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("non-finite logits")
    return int(np.argmax(logits))


def competitor(original_class: int) -> int:
    if original_class not in (0, 1):
        raise ValueError(f"binary classes only, got {original_class}")
    return 1 - original_class


@dataclass(frozen=True)
class TargetDefinition:
    policy: TargetPolicy | None
    margin: float | None = None
    prediction_tie_rule: str = PREDICTION_TIE_RULE

    def __post_init__(self) -> None:
        if self.prediction_tie_rule != PREDICTION_TIE_RULE:
            raise ValueError(f"only tie rule {PREDICTION_TIE_RULE!r} is implemented")
        if self.policy is TargetPolicy.POSITIVE_MARGIN:
            if self.margin is None or not self.margin > 0:
                raise ValueError("positive_margin policy needs margin > 0")
        elif self.margin is not None:
            raise ValueError(f"margin must be null for policy {self.policy}")

    @classmethod
    def from_config(cls, cfg: dict) -> "TargetDefinition":
        policy = cfg.get("policy")
        return cls(
            policy=TargetPolicy(policy) if policy is not None else None,
            margin=cfg.get("margin"),
            prediction_tie_rule=cfg.get("prediction_tie_rule", PREDICTION_TIE_RULE),
        )

    @property
    def resolved(self) -> bool:
        return self.policy is not None

    @property
    def target_id(self) -> str:
        if self.policy is None:
            return "unresolved"
        if self.policy is TargetPolicy.POSITIVE_MARGIN:
            return f"{self.policy.value}-m{self.margin:g}-tie_{self.prediction_tie_rule}"
        return f"{self.policy.value}-tie_{self.prediction_tie_rule}"

    def require_resolved(self) -> None:
        if not self.resolved:
            raise UnresolvedTargetError(
                "target.policy is unresolved (decision D001 in research/log.md); "
                "target-dependent computations are disabled until it is decided"
            )


@dataclass(frozen=True)
class TargetEvaluation:
    target_id: str
    original_class: int
    competitor_class: int
    logits: tuple[float, float]
    predicted_class: int
    score_gap: float
    score_condition_satisfied: bool
    prediction_changed: bool
    target_satisfied: bool


def evaluate_target(target: TargetDefinition, logits: np.ndarray, original_class: int) -> TargetEvaluation:
    target.require_resolved()
    logits = np.asarray(logits, dtype=np.float64)
    pred = predict(logits)
    j = competitor(original_class)
    gap = float(logits[j] - logits[original_class])

    threshold = target.margin if target.policy is TargetPolicy.POSITIVE_MARGIN else 0.0
    score_ok = gap >= threshold
    changed = pred != original_class
    satisfied = changed if target.policy is TargetPolicy.STRICT_PREDICTION_CHANGE else score_ok

    return TargetEvaluation(
        target_id=target.target_id,
        original_class=original_class,
        competitor_class=j,
        logits=(float(logits[0]), float(logits[1])),
        predicted_class=pred,
        score_gap=gap,
        score_condition_satisfied=bool(score_ok),
        prediction_changed=bool(changed),
        target_satisfied=bool(satisfied),
    )
