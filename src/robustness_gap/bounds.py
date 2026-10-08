from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .models import AffineReLUNetwork

DOMAIN_LO = 0.0
DOMAIN_HI = 1.0


def input_box(x: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if eps < 0:
        raise ValueError("eps must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(DOMAIN_LO, x - eps), np.minimum(DOMAIN_HI, x + eps)


def domain_box(dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    return np.full(dim, DOMAIN_LO), np.full(dim, DOMAIN_HI)


def point_cover_radius(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.maximum(x - DOMAIN_LO, DOMAIN_HI - x)))


def affine_interval(
    W: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray, pad: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(lo > hi):
        raise ValueError("empty input interval")
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    z_lo = W_pos @ lo + W_neg @ hi + b - pad
    z_hi = W_pos @ hi + W_neg @ lo + b + pad
    return z_lo, z_hi


class ReluState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    UNSTABLE = "unstable"


def relu_state(l: float, u: float) -> ReluState:
    if l > u:
        raise ValueError(f"invalid bounds l={l} > u={u}")
    if u <= 0:
        return ReluState.INACTIVE
    if l >= 0:
        return ReluState.ACTIVE
    return ReluState.UNSTABLE


@dataclass(frozen=True)
class LayerBounds:
    pre_lo: np.ndarray
    pre_hi: np.ndarray

    @property
    def states(self) -> list[ReluState]:
        return [relu_state(l, u) for l, u in zip(self.pre_lo, self.pre_hi)]


@dataclass(frozen=True)
class NetworkBounds:
    input_lo: np.ndarray
    input_hi: np.ndarray
    hidden: tuple[LayerBounds, ...]
    logits_lo: np.ndarray
    logits_hi: np.ndarray
    method: str = "interval"

    def unstable_count(self) -> int:
        return sum(s is ReluState.UNSTABLE for layer in self.hidden for s in layer.states)


def interval_bounds(
    net: AffineReLUNetwork, lo: np.ndarray, hi: np.ndarray, pad: float = 0.0
) -> NetworkBounds:
    a_lo, a_hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    hidden = []
    for W, b in zip(net.weights[:-1], net.biases[:-1]):
        z_lo, z_hi = affine_interval(W, b, a_lo, a_hi, pad)
        hidden.append(LayerBounds(z_lo, z_hi))
        a_lo, a_hi = np.maximum(z_lo, 0.0), np.maximum(z_hi, 0.0)
    out_lo, out_hi = affine_interval(net.weights[-1], net.biases[-1], a_lo, a_hi, pad)
    return NetworkBounds(np.asarray(lo), np.asarray(hi), tuple(hidden), out_lo, out_hi)


def triangle_upper(z: float, l: float, u: float) -> float:
    if not (l < 0 < u):
        raise ValueError("triangle relaxation is only used for unstable ReLUs")
    return u * (z - l) / (u - l)
