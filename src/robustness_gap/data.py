from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import records

GENERATOR_ID = "two_moons_numpy_v1"
SPLITS = ("train", "val", "test")


def generate_raw(n_total: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_total % 2:
        raise ValueError("n_total must be even")
    n = n_total // 2
    t = np.linspace(0.0, np.pi, n)
    c0 = np.stack([np.cos(t), np.sin(t)], axis=1)
    c1 = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([c0, c1]).astype(np.float64)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    rng = np.random.default_rng(seed)
    X = X + rng.normal(0.0, noise, size=X.shape)
    return X, y


def apply_transform(X_raw: np.ndarray, tcfg: dict) -> np.ndarray:
    if tcfg["method"] != "fixed_isotropic_affine":
        raise ValueError(f"unknown transform {tcfg['method']!r}")
    center = np.asarray(tcfg["source_center"], dtype=np.float64)
    half = float(tcfg["source_half_width"])
    X = (X_raw - center) / (2.0 * half) + 0.5
    outside = np.any((X < 0.0) | (X > 1.0), axis=1)
    if outside.any():
        if tcfg["out_of_domain_policy"] == "error":
            raise ValueError(
                f"{int(outside.sum())} points fall outside D=[0,1]^2 under the declared transform; "
                "revise the transform or noise in a recorded decision rather than clipping"
            )
        raise ValueError(f"unknown out_of_domain_policy {tcfg['out_of_domain_policy']!r}")
    return X


def stratified_split(y: np.ndarray, sizes: dict[str, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.empty(len(y), dtype=object)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        start = 0
        for name in SPLITS:
            k = sizes[name] // 2
            split[idx[start:start + k]] = name
            start += k
    if any(s is None for s in split):
        raise AssertionError("unassigned samples")
    return split.astype(str)


def dataset_id(dcfg: dict, domain_cfg: dict) -> str:
    return f"moons-{records.sha256_json({'dataset': dcfg, 'domain': domain_cfg})[:12]}"


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    split: np.ndarray

    def subset(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.split == name
        return self.ids[m], self.X[m], self.y[m]


def generate_dataset(cfg: dict, data_root: Path) -> Path:
    dcfg, domain_cfg = cfg["dataset"], cfg["domain"]
    if dcfg["generator"] != GENERATOR_ID:
        raise ValueError(f"unknown generator {dcfg['generator']!r}")
    did = dataset_id(dcfg, domain_cfg)
    out = data_root / did
    if (out / "manifest.json").exists():
        verify_dataset(out)
        return out

    X_raw, y = generate_raw(dcfg["n_total"], dcfg["noise"], dcfg["generation_seed"])
    X = apply_transform(X_raw, dcfg["transform"])
    sizes = {"train": dcfg["n_train"], "val": dcfg["n_val"], "test": dcfg["n_test"]}
    split = stratified_split(y, sizes, dcfg["split_seed"])
    ids = np.array([f"s{i:05d}" for i in range(len(y))])

    out.mkdir(parents=True, exist_ok=False)
    arrays = out / "arrays.npz"
    np.savez(arrays, ids=ids, X_raw=X_raw, X=X, y=y, split=split)
    csv_lines = ["sample_id,split,label,x0,x1,raw0,raw1"] + [
        f"{i},{s},{l},{a[0]!r},{a[1]!r},{r[0]!r},{r[1]!r}"
        for i, s, l, a, r in zip(ids, split, y, X.tolist(), X_raw.tolist())
    ]
    records.atomic_write_bytes(out / "samples.csv", ("\n".join(csv_lines) + "\n").encode())

    counts = {s: {str(c): int(((split == s) & (y == c)).sum()) for c in (0, 1)} for s in SPLITS}
    manifest = records.make_record("dataset", {
        "dataset_id": did,
        "generator": GENERATOR_ID,
        "numpy_version": np.__version__,
        "config": dcfg,
        "domain": domain_cfg,
        "class_counts": counts,
        "raw_min": X_raw.min(axis=0), "raw_max": X_raw.max(axis=0),
        "normalized_min": X.min(axis=0), "normalized_max": X.max(axis=0),
        "files": {p.name: records.sha256_file(p) for p in (arrays, out / "samples.csv")},
    }, input_hash=records.sha256_json({"dataset": dcfg, "domain": domain_cfg}))
    records.atomic_write_json(out / "manifest.json", manifest)
    return out


def verify_dataset(path: Path) -> dict:
    manifest = records.read_json(path / "manifest.json")
    for name, digest in manifest["files"].items():
        if records.sha256_file(path / name) != digest:
            raise ValueError(f"{path / name}: hash mismatch with manifest")
    return manifest


def load_dataset(path: Path) -> Dataset:
    manifest = verify_dataset(path)
    z = np.load(path / "arrays.npz", allow_pickle=False)
    return Dataset(manifest["dataset_id"], z["ids"], z["X"], z["y"], z["split"])
