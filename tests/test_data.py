import numpy as np
import pytest

from robustness_gap.data import apply_transform, generate_dataset, load_dataset


@pytest.fixture
def dataset(tmp_path, pilot_cfg):
    return load_dataset(generate_dataset(pilot_cfg, tmp_path))


def test_sizes_balance_and_domain(dataset, pilot_cfg):
    d = pilot_cfg["dataset"]
    for name, n in (("train", d["n_train"]), ("val", d["n_val"]), ("test", d["n_test"])):
        _, X, y = dataset.subset(name)
        assert len(y) == n and (y == 0).sum() == n // 2
    assert dataset.X.min() >= 0 and dataset.X.max() <= 1
    assert len(set(dataset.ids)) == len(dataset.ids)


def test_regeneration_is_identical_and_never_overwrites(tmp_path, pilot_cfg):
    p1 = generate_dataset(pilot_cfg, tmp_path)
    before = (p1 / "arrays.npz").read_bytes()
    p2 = generate_dataset(pilot_cfg, tmp_path)
    assert p1 == p2 and (p2 / "arrays.npz").read_bytes() == before


def test_tampered_dataset_detected(tmp_path, pilot_cfg):
    p = generate_dataset(pilot_cfg, tmp_path)
    with open(p / "samples.csv", "a") as f:
        f.write("tamper\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(p)


def test_out_of_domain_is_an_error_not_a_clip():
    t = {"method": "fixed_isotropic_affine", "source_center": [0, 0], "source_half_width": 1.0,
         "out_of_domain_policy": "error"}
    with pytest.raises(ValueError, match="outside D"):
        apply_transform(np.array([[0.0, 0.0], [5.0, 0.0]]), t)
