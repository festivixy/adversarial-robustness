from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .data import Dataset

STYLE = {
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.prop_cycle": matplotlib.cycler(color=["black"]),
    "savefig.bbox": "tight",
}
MARKERS = {0: "o", 1: "^"}


def save(fig, out_stem: Path) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    for p in paths:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None)
    plt.close(fig)
    return paths


def plot_dataset(ds: Dataset, out_stem: Path, split: str = "train") -> list[Path]:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        _, X, y = ds.subset(split)
        for cls in (0, 1):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=10, marker=MARKERS[cls],
                       facecolors="none" if cls == 0 else "black", edgecolors="black",
                       linewidths=0.6, label=f"class {cls}")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
               title=f"{ds.dataset_id} ({split})")
        ax.legend(loc="upper right", frameon=True)
        return save(fig, out_stem)
