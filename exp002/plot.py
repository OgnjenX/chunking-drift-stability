"""Descriptive development figures; no confirmation or stability inference."""

import json
from pathlib import Path

import matplotlib
import numpy as np

from .develop_probe import DEV_LONG, DEV_OLD

# Select a headless backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports,wrong-import-order


def main():
    """Plot development performance, activity, and mechanism diagnostics."""
    root = Path("exp002/results/development-calibration")
    rows = json.loads((root / "trials.json").read_text())
    d = json.loads((root / "diagnostics/trials.json").read_text())
    codes = np.load(root / "diagnostics/s101_codes.npz")["learned"][:, 8:]
    temporal = np.load(root / "diagnostics/s101_temporal.npz")
    colors = {"field": "#a63a50", "gru": "#218380", "nonchunk_wm": "#777777", "store2": "#b7950b"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    im = axes[0, 0].imshow(codes, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[0, 0].set(
        yticks=np.arange(20),
        yticklabels=DEV_OLD + DEV_LONG,
        xlabel="Potential chunk cell",
        title="Learned field activity: development seed 101",
    )
    fig.colorbar(im, ax=axes[0, 0], label="Rectified activity")
    levels = ["clean", "light", "medium", "partial", "hard"]
    for model, color in colors.items():
        per_seed = np.array(
            [
                [
                    np.mean(
                        [
                            r["exact"]
                            for r in rows
                            if r["model"] == model
                            and r["difficulty"] == level
                            and r["seed"] == seed
                        ]
                    )
                    for level in levels
                ]
                for seed in [101, 102, 103]
            ]
        )
        for values in per_seed:
            axes[0, 1].plot(range(5), values, color=color, alpha=0.25, lw=0.8)
        axes[0, 1].plot(range(5), per_seed.mean(axis=0), color=color, marker="o", label=model)
    axes[0, 1].axhspan(0.45, 0.85, color="grey", alpha=0.1, label="Desired common headroom")
    axes[0, 1].set(
        xticks=range(5),
        xticklabels=levels,
        ylim=(-0.02, 1.02),
        ylabel="Exact entire-list accuracy",
        title="Difficulty calibration failed (three seeds)",
    )
    axes[0, 1].legend(fontsize=8)
    groups = [list("ABCD"), ["AC", "CA", "AD", "DA", "BD", "DB"], DEV_LONG]
    for k, model in enumerate(["field", "nonchunk_wm", "gru"]):
        values = [
            np.mean(
                [
                    r["exact"]
                    for r in d
                    if r["model"] == model
                    and r["intervention"] == "intact"
                    and r["condition"] == "training_matched"
                    and r["sequence"] in group
                ]
            )
            for group in groups
        ]
        axes[1, 0].bar(
            np.arange(3) + (k - 1) * 0.25, values, width=0.25, color=colors[model], label=model
        )
    axes[1, 0].set(
        xticks=range(3),
        xticklabels=["Old items", "Old short lists", "New longer lists"],
        ylim=(0, 1.05),
        ylabel="Exact accuracy",
        title="Training-matched probes after longer-list learning",
    )
    axes[1, 0].legend(fontsize=8)
    variants = ["intact", "feedback_off", "competition_off", "chunks_off", "chunk_lesion"]
    values = [
        np.mean(
            [
                r["exact"]
                for r in d
                if r["model"] == "field"
                and r["intervention"] == v
                and r["condition"] == "training_matched"
            ]
        )
        for v in variants
    ]
    axes[1, 1].bar(range(5), values, color=["#a63a50"] + ["#777777"] * 4)
    axes[1, 1].set(
        xticks=range(5),
        xticklabels=[
            "Intact",
            "Feedback\noff",
            "Competition\noff",
            "Chunks\noff",
            "25% chunk\nlesion",
        ],
        ylim=(0, 1),
        ylabel="Exact accuracy",
        title="Ablations affect output; selective chunks still absent",
    )
    fig.suptitle("EXP002 exploratory development only · prerequisite failure · no drift result")
    fig.savefig(root / "development-overview.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    t = np.arange(len(temporal["stimulus"])) * float(temporal["dt"])
    for i, label in enumerate("ABCD"):
        axes[0].plot(t, temporal["stimulus"][:, i], label=label)
        axes[1].plot(t, temporal["trace"][:, 4 + i], label=label)
    axes[0].set(
        ylabel="Input pulse",
        title="Development sequence ACDB: successive items and retention delay",
    )
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set(ylabel="Lower Y activity")
    axes[1].legend(ncol=4, fontsize=8)
    c = temporal["trace"][:, 8:]
    axes[2].plot(t, c.max(axis=1), label="Maximum chunk activity", color="#a63a50")
    axes[2].plot(t, c.mean(axis=1), label="Mean across chunk cells", color="#777777")
    axes[2].set(ylabel="Higher activity", xlabel="Model time (no biological-unit claim)")
    axes[2].legend(fontsize=8)
    fig.savefig(root / "temporal-dynamics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
