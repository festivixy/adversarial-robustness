from __future__ import annotations

from dataclasses import dataclass

from .records import Status


@dataclass
class AttackResult:
    U: float | None
    attack_status: Status
    candidate: list[float] | None
    candidate_logits: tuple[float, float] | None
    candidate_label: int | None
    recomputed_distance: float | None
    valid_domain: bool | None
    valid_target: bool | None
    implementation_id: str
    seed: int
    restarts: int
    iterations: int
    elapsed_seconds: float | None


def attack(*args, **kwargs) -> AttackResult:
    raise NotImplementedError("attack adapter is blocked on decisions D001 and D003")
