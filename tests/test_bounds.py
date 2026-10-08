import numpy as np
import pytest

from robustness_gap.bounds import (ReluState, affine_interval, input_box, interval_bounds,
                                   point_cover_radius, relu_state, triangle_upper)
from robustness_gap.models import AffineReLUNetwork

from conftest import linear_net


def test_input_box_clipped_to_domain():
    lo, hi = input_box(np.array([0.05, 0.9]), 0.1)
    np.testing.assert_allclose(lo, [0.0, 0.8])
    np.testing.assert_allclose(hi, [0.15, 1.0])


def test_affine_interval_hand_calculated():
    lo, hi = affine_interval(np.array([[2.0, -3.0]]), np.array([1.0]),
                             np.array([0.0, 0.5]), np.array([1.0, 2.0]))
    assert lo[0] == pytest.approx(2 * 0 - 3 * 2 + 1)
    assert hi[0] == pytest.approx(2 * 1 - 3 * 0.5 + 1)


def test_affine_interval_rejects_empty_box():
    with pytest.raises(ValueError):
        affine_interval(np.eye(2), np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))


@pytest.mark.parametrize("l,u,state", [
    (-1.0, 1.0, ReluState.UNSTABLE),
    (-1.0, 0.0, ReluState.INACTIVE),
    (0.0, 1.0, ReluState.ACTIVE),
    (0.0, 0.0, ReluState.INACTIVE),
    (-2.0, -1.0, ReluState.INACTIVE),
    (1.0, 2.0, ReluState.ACTIVE),
])
def test_relu_states(l, u, state):
    assert relu_state(l, u) is state


def test_triangle_upper_face():
    assert triangle_upper(-1.0, -1.0, 1.0) == pytest.approx(0.0)
    assert triangle_upper(1.0, -1.0, 1.0) == pytest.approx(1.0)
    assert triangle_upper(0.0, -1.0, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        triangle_upper(0.0, 0.0, 1.0)


def test_point_cover_radius():
    assert point_cover_radius(np.array([0.4, 0.6])) == pytest.approx(0.6)
    assert point_cover_radius(np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_interval_bounds_contain_sampled_outputs():
    rng = np.random.default_rng(1)
    Ws = (rng.normal(size=(8, 2)), rng.normal(size=(8, 8)), rng.normal(size=(2, 8)))
    bs = (rng.normal(size=8), rng.normal(size=8), rng.normal(size=2))
    net = AffineReLUNetwork(Ws, bs)
    x = np.array([0.3, 0.7])
    lo, hi = input_box(x, 0.05)
    nb = interval_bounds(net, lo, hi)
    for p in rng.uniform(lo, hi, size=(2000, 2)):
        pre, out = net.forward_trace(p)
        for layer, z in zip(nb.hidden, pre):
            assert np.all(z >= layer.pre_lo - 1e-12) and np.all(z <= layer.pre_hi + 1e-12)
        assert np.all(out >= nb.logits_lo - 1e-12) and np.all(out <= nb.logits_hi + 1e-12)


def test_linear_boundary_distance_example():
    net = linear_net((1.0, 0.0), b1=-0.5)
    x = np.array([0.4, 0.6])
    gap_hi = lambda eps: interval_bounds(net, *input_box(x, eps)).logits_hi[1]
    assert gap_hi(0.0999) < 0
    assert gap_hi(0.1) == pytest.approx(0.0, abs=1e-15)
    assert gap_hi(0.1001) > 0
