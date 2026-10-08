from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

INPUT_DIM = 2
N_LOGITS = 2

PROPOSED_ARCHITECTURES: dict[str, tuple[tuple[int, ...], int, int]] = {
    "mlp1": ((36,), 182, 36),
    "mlp2": ((11, 11), 189, 22),
    "mlp3": ((8, 8, 8), 186, 24),
}


def layer_sizes(hidden_widths: tuple[int, ...] | list[int]) -> list[int]:
    return [INPUT_DIM, *hidden_widths, N_LOGITS]


def parameter_count(hidden_widths) -> int:
    sizes = layer_sizes(hidden_widths)
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))


def relu_count(hidden_widths) -> int:
    return int(sum(hidden_widths))


def check_architecture_table(architectures: dict[str, list[int]]) -> list[str]:
    problems = []
    for arch_id, widths in architectures.items():
        expected = PROPOSED_ARCHITECTURES.get(arch_id)
        if expected is None:
            problems.append(f"{arch_id}: not in proposed table")
            continue
        exp_widths, exp_params, exp_relus = expected
        if tuple(widths) != exp_widths:
            problems.append(f"{arch_id}: widths {widths} != proposed {list(exp_widths)}")
        if parameter_count(widths) != exp_params:
            problems.append(f"{arch_id}: {parameter_count(widths)} params != {exp_params}")
        if relu_count(widths) != exp_relus:
            problems.append(f"{arch_id}: {relu_count(widths)} ReLUs != {exp_relus}")
    return problems


def build_mlp(hidden_widths, dtype: torch.dtype = torch.float32) -> nn.Sequential:
    sizes = layer_sizes(hidden_widths)
    layers: list[nn.Module] = []
    for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(i, o, bias=True, dtype=dtype))
        if k < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class AffineReLUNetwork:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError("need matching, non-empty weight/bias lists")
        prev = INPUT_DIM
        for W, b in zip(self.weights, self.biases):
            if W.ndim != 2 or W.shape[1] != prev or b.shape != (W.shape[0],):
                raise ValueError(f"inconsistent layer shapes {W.shape}, {b.shape}")
            prev = W.shape[0]
        if prev != N_LOGITS:
            raise ValueError(f"network must output {N_LOGITS} logits")

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        return tuple(W.shape[0] for W in self.weights[:-1])

    @classmethod
    def from_torch(cls, model: nn.Sequential) -> "AffineReLUNetwork":
        for m in model:
            if not isinstance(m, (nn.Linear, nn.ReLU)):
                raise TypeError(f"unsupported layer {type(m).__name__}")
        linears = [m for m in model if isinstance(m, nn.Linear)]
        return cls(
            weights=tuple(m.weight.detach().cpu().to(torch.float64).numpy().copy() for m in linears),
            biases=tuple(m.bias.detach().cpu().to(torch.float64).numpy().copy() for m in linears),
        )

    def forward_trace(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        a = np.asarray(x, dtype=np.float64)
        if a.shape != (INPUT_DIM,):
            raise ValueError(f"expected input shape ({INPUT_DIM},), got {a.shape}")
        pre = []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = W @ a + b
            pre.append(z)
            a = np.maximum(z, 0.0)
        return pre, self.weights[-1] @ a + self.biases[-1]

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.forward_trace(x)[1]
