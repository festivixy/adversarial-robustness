from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "torch", "pandas", "matplotlib", "pyyaml", "highspy")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def code_identity(repo: Path) -> dict:
    rev = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain"], repo)
    return {
        "code_revision": rev or "no_commit",
        "dirty_tree": bool(status) if status is not None else None,
    }


def code_hash() -> str:
    package = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        h.update(path.relative_to(package).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment(repo: Path) -> dict:
    import torch

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "packages": package_versions(),
        "code_hash": code_hash(),
        **code_identity(repo),
    }
