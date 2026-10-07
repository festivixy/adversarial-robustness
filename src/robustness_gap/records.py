"""Record formats, status vocabulary, hashing, and safe writing to disk.

Every result produced by the pipeline is saved as a JSON record that carries a schema version, a
kind, and an input hash. The input hash is a SHA-256 fingerprint of everything that produced the
record, such as the network weights, the target definition, and the attack budget. When the pipeline
is rerun it reuses a saved record only if the fingerprint still matches, and if the fingerprint has
changed it stops with a StaleRecordError instead of quietly reusing or overwriting old work, because
changed inputs call for a new experiment id.

Records are written atomically. The data first goes to a temporary file in the same directory, is
flushed to disk, and is then renamed over the destination in a single step, so an interrupted run
never leaves a half-written file and can simply be resumed.

Standard JSON has no representation for NaN or infinity, and silently writing them would let a
missing measurement masquerade as a number. The writer therefore rejects them outright, and absent
values are stored as null together with an explicit status from the Status vocabulary. Each adapter
is allowed only a subset of those statuses, which ALLOWED_STATUSES lists, and check_status enforces.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


class Status(str, Enum):
    SUCCESS = "success"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    NO_POSITIVE_CERTIFICATE = "no_positive_certificate"
    TIMEOUT_WITH_BOUNDS = "timeout_with_bounds"
    TIMEOUT_WITHOUT_BOUNDS = "timeout_without_bounds"
    RESOLVED = "resolved"
    INFEASIBLE_TARGET = "infeasible_target"
    NUMERICAL_FAILURE = "numerical_failure"
    INVALID_CANDIDATE = "invalid_candidate"
    INCONSISTENT_BOUNDS = "inconsistent_bounds"
    NOT_EVALUATED = "not_evaluated"
    CANCELLED = "cancelled"
    UNBOUNDED_RADIUS = "unbounded_radius"
    UNAVAILABLE = "unavailable"


ALLOWED_STATUSES: dict[str, frozenset[Status]] = {
    "certificate": frozenset({
        Status.SUCCESS, Status.NO_POSITIVE_CERTIFICATE, Status.NUMERICAL_FAILURE,
        Status.NOT_EVALUATED, Status.CANCELLED,
    }),
    "attack": frozenset({
        Status.SUCCESS, Status.NO_VALID_CANDIDATE, Status.INVALID_CANDIDATE,
        Status.NUMERICAL_FAILURE, Status.NOT_EVALUATED, Status.CANCELLED,
    }),
    "reference": frozenset({
        Status.RESOLVED, Status.TIMEOUT_WITH_BOUNDS, Status.TIMEOUT_WITHOUT_BOUNDS,
        Status.INFEASIBLE_TARGET, Status.NUMERICAL_FAILURE, Status.INVALID_CANDIDATE,
        Status.NOT_EVALUATED, Status.CANCELLED,
    }),
    "derived": frozenset({
        Status.SUCCESS, Status.INCONSISTENT_BOUNDS, Status.UNAVAILABLE, Status.NOT_EVALUATED,
    }),
}


class StaleRecordError(RuntimeError):
    def __init__(self, path, stored, expected):
        super().__init__(
            f"{path}: stored input_hash {stored} != current {expected}; inputs changed, "
            "so use a new experiment id instead of reusing or overwriting this record"
        )


def check_status(adapter: str, status: Status) -> Status:
    if status not in ALLOWED_STATUSES[adapter]:
        raise ValueError(f"status {status.value!r} not allowed for adapter {adapter!r}")
    return status


def _default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def dumps(obj: Any, *, canonical: bool = False) -> str:
    if canonical:
        return json.dumps(obj, default=_default, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(obj, default=_default, allow_nan=False, indent=2)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(dumps(obj, canonical=True).encode())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, (dumps(obj) + "\n").encode())


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_new_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    if path.exists():
        if read_json(path) == json.loads(dumps(obj)):
            return
        raise FileExistsError(f"refusing to overwrite existing record {path}")
    atomic_write_json(path, obj)


def make_record(kind: str, payload: dict, input_hash: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "input_hash": input_hash, **payload}


def load_if_current(path: str | Path, input_hash: str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    rec = read_json(path)
    if rec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema version {rec.get('schema_version')} != {SCHEMA_VERSION}")
    if rec.get("input_hash") != input_hash:
        raise StaleRecordError(path, rec.get("input_hash"), input_hash)
    return rec
