from pathlib import Path

import numpy as np
import pytest

from robustness_gap.config import load_config
from robustness_gap.models import AffineReLUNetwork

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def pilot_cfg():
    return load_config(REPO / "configs" / "pilot.yaml")[1]


def linear_net(w, b0=0.0, b1=0.0):
    W = np.array([[0.0, 0.0], list(w)], dtype=np.float64)
    return AffineReLUNetwork((W,), (np.array([b0, b1], dtype=np.float64),))
