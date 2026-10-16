"""Descriptive unit-consistency figures; initial and learned filters side by side."""

import json
from pathlib import Path

import matplotlib
import numpy as np

from ..develop_probe import DEV_LONG, DEV_OLD

# Select a headless backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports,wrong-import-order


def main():
    """Plot source-unit development results and untrained-filter controls."""
    root = Path("exp002/results/unit-consistency-calibration")
    z = np.load(root / "s101_codes.npz")
    decision = json.loads((root / "decision.json").read_text())
    codes = json.loads((root / "mechanisms.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, key, title in [
        (axes[0, 0], "learned", "Learned filters, seed 101"),
        (axes[0, 1], "untrained", "Untrained hidden filters + learned task head"),
    ]:
        im = ax.imshow(z[key][:, 8:], aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set(
            yticks=np.arange(20),
            yticklabels=DEV_OLD + DEV_LONG,
            xlabel="Potential field cell",
            title=title,
        )
        fig.colorbar(im, ax=ax, label="Activity")
    rows = decision["performance"]
    for key, color, label in [
        ("field", "#a63a50", "Learned filters"),
        ("frozen_filters", "#8d6e63", "Untrained filters"),
        ("gru", "#218380", "Ordinary GRU"),
    ]:
        axes[1, 0].plot(
            range(len(rows)), [r[key] for r in rows], marker="o", color=color, label=label
        )
    axes[1, 0].axhspan(0.45, 0.85, color="grey", alpha=0.1)
    axes[1, 0].set(
        xticks=range(len(rows)),
        xticklabels=["Clean", "Light", "Medium", "Partial", "Hard", "Matched"],
        ylim=(-0.02, 1.03),
        ylabel="Exact list accuracy",
        title="Learning hidden filters adds no measured advantage",
    )
    axes[1, 0].legend(fontsize=8)
    xx = np.arange(3)
    axes[1, 1].bar(
        xx - 0.18,
        [r["trained_winners"] for r in codes],
        0.36,
        color="#a63a50",
        label="Learned filters",
    )
    axes[1, 1].bar(
        xx + 0.18,
        [r["untrained_filter_winners"] for r in codes],
        0.36,
        color="#8d6e63",
        label="Untrained filters",
    )
    axes[1, 1].set(
        xticks=xx,
        xticklabels=["101", "102", "103"],
        ylim=(0, 21),
        xlabel="Development seed",
        ylabel="Distinct winning cells / 20 lists",
        title="Selectivity is already present before filter learning",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "EXP002 second development variant · explicit source-unit conversion · no confirmation"
    )
    fig.savefig(root / "unit-consistency-overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
