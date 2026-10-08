import numpy as np
import pytest
import torch

from robustness_gap.models import (PROPOSED_ARCHITECTURES, AffineReLUNetwork, build_mlp,
                                   check_architecture_table, parameter_count, relu_count)


@pytest.mark.parametrize("arch", PROPOSED_ARCHITECTURES)
def test_parameter_and_relu_counts(arch):
    widths, params, relus = PROPOSED_ARCHITECTURES[arch]
    assert parameter_count(widths) == params
    assert relu_count(widths) == relus
    model = build_mlp(widths)
    assert sum(p.numel() for p in model.parameters()) == params


def test_config_table_matches(pilot_cfg):
    assert check_architecture_table(pilot_cfg["model"]["architectures"]) == []


def test_mismatch_detected():
    assert check_architecture_table({"mlp2": [12, 11]})


@pytest.mark.parametrize("arch", PROPOSED_ARCHITECTURES)
def test_torch_and_numpy_logits_agree(arch):
    torch.manual_seed(0)
    model = build_mlp(PROPOSED_ARCHITECTURES[arch][0], torch.float32).eval()
    net = AffineReLUNetwork.from_torch(model)
    rng = np.random.default_rng(0)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]], dtype=np.float64)
    pts = np.concatenate([rng.random((200, 2)), corners])
    with torch.no_grad():
        t32 = model(torch.tensor(pts, dtype=torch.float32)).numpy().astype(np.float64)
    np64 = np.stack([net.logits(p) for p in pts])
    np.testing.assert_allclose(np64, t32, atol=1e-5)
    model64 = build_mlp(PROPOSED_ARCHITECTURES[arch][0], torch.float64)
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    with torch.no_grad():
        t64 = model64(torch.tensor(pts, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(np64, t64, atol=1e-12)


def test_forward_trace_activations():
    W1 = np.array([[1.0, 0.0], [0.0, -1.0]]); b1 = np.array([0.0, 0.5])
    W2 = np.eye(2); b2 = np.zeros(2)
    net = AffineReLUNetwork((W1, W2), (b1, b2))
    pre, out = net.forward_trace(np.array([0.2, 0.7]))
    np.testing.assert_allclose(pre[0], [0.2, -0.2])
    np.testing.assert_allclose(out, [0.2, 0.0])
