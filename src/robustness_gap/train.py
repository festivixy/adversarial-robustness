from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import records
from .collection import model_id
from .config import scientific_hash
from .data import Dataset
from .models import AffineReLUNetwork, build_mlp, parameter_count, relu_count

DTYPES = {"float32": torch.float32, "float64": torch.float64}
SUPPORTED = ("cross_entropy", "adam", "best_val_loss_earliest_tie", "cpu")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(X).argmax(dim=1) == y).float().mean())


def split_tensors(ds: Dataset, name: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    _, X, y = ds.subset(name)
    return torch.tensor(X, dtype=dtype), torch.tensor(y, dtype=torch.long)


def train_one(cfg: dict, ds: Dataset, dataset_hash: str, arch_id: str, seed: int,
              out_dir: Path, provenance: dict) -> dict:
    tcfg, mcfg = cfg["training"], cfg["model"]
    settings = (tcfg["loss"], tcfg["optimizer"], tcfg["checkpoint_rule"], tcfg["device"])
    if settings != SUPPORTED:
        raise NotImplementedError(f"only {SUPPORTED} is implemented, got {settings}")
    widths = mcfg["architectures"][arch_id]
    dtype = DTYPES[mcfg["train_dtype"]]
    input_hash = records.sha256_json({
        "science": scientific_hash(cfg), "dataset_hash": dataset_hash, "arch": arch_id, "seed": seed,
    })
    record_path = out_dir / "record.json"
    cached = records.load_if_current(record_path, input_hash)
    if cached is not None:
        return cached

    torch.set_num_threads(tcfg["torch_threads"])
    gen = seed_everything(seed)
    model = build_mlp(widths, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    Xtr, ytr = split_tensors(ds, "train", dtype)
    Xva, yva = split_tensors(ds, "val", dtype)

    best_loss, best_epoch, best_state = float("inf"), None, None
    curve = ["epoch,train_loss,val_loss,val_accuracy"]
    status = "success"
    t0 = time.perf_counter()
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        total = 0.0
        for i in range(0, len(Xtr), tcfg["batch_size"]):
            idx = perm[i:i + tcfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva))
        if not np.isfinite(val_loss):
            status = "numerical_failure"
            break
        curve.append(f"{epoch},{total / len(Xtr)!r},{val_loss!r},{_accuracy(model, Xva, yva)!r}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    records.atomic_write_bytes(out_dir / "curve.csv", ("\n".join(curve) + "\n").encode())
    ckpt_path, ckpt_hash, accs = None, None, {}
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        ckpt_path = out_dir / "checkpoint.pt"
        tmp = out_dir / ".checkpoint.pt.tmp"
        torch.save({"hidden_widths": list(widths), "dtype": mcfg["train_dtype"],
                    "state_dict": best_state}, tmp)
        tmp.replace(ckpt_path)
        ckpt_hash = records.sha256_file(ckpt_path)
        accs = {name: _accuracy(model, *split_tensors(ds, name, dtype)) for name in ("train", "val", "test")}

    rec = records.make_record("model", {
        "model_id": model_id(arch_id, seed),
        "architecture_id": arch_id,
        "hidden_widths": list(widths),
        "parameter_count": parameter_count(widths),
        "relu_count": relu_count(widths),
        "training_seed": seed,
        "dataset_id": ds.dataset_id,
        "dataset_hash": dataset_hash,
        "checkpoint_path": ckpt_path.relative_to(out_dir).as_posix() if ckpt_path else None,
        "checkpoint_hash": ckpt_hash,
        "config_hash": scientific_hash(cfg),
        "code_revision": provenance.get("code_revision"),
        "code_hash": provenance.get("code_hash"),
        "dirty_tree": provenance.get("dirty_tree"),
        "machine": provenance.get("machine"),
        "cpu_model": provenance.get("cpu_model"),
        "dtype": mcfg["train_dtype"],
        "device": tcfg["device"],
        "torch_threads": tcfg["torch_threads"],
        "epochs_run": len(curve) - 1,
        "selected_epoch": best_epoch,
        "best_val_loss": best_loss if best_state is not None else None,
        "train_accuracy": accs.get("train"),
        "validation_accuracy": accs.get("val"),
        "test_accuracy": accs.get("test"),
        "training_seconds": elapsed,
        "status": status,
    }, input_hash)
    records.atomic_write_json(record_path, rec)
    return rec


def load_network(record: dict, model_dir: Path) -> tuple[nn.Sequential, AffineReLUNetwork]:
    path = model_dir / record["checkpoint_path"]
    if records.sha256_file(path) != record["checkpoint_hash"]:
        raise ValueError(f"{path}: checkpoint hash mismatch")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_mlp(ckpt["hidden_widths"], DTYPES[ckpt["dtype"]])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, AffineReLUNetwork.from_torch(model)
