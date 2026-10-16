"""Descriptive post-run plots; no changes to registered decisions or training."""

import json
from pathlib import Path

import matplotlib
import numpy as np

# Select a headless backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports


def main():
    """Plot descriptive retention, geometry, and intervention diagnostics."""
    root = Path("results/registered-v1")
    rows = json.loads((root / "metrics.json").read_text())
    colors = {"flat": "#777777", "dense": "#218380", "competitive": "#a63a50"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ts = [0, 10, 25, 50, 100]
    for kind, color in colors.items():
        a = [
            r
            for r in rows
            if r["kind"] == kind
            and r["noise"] == 0.08
            and r["intervention"] == "intact"
            and r["plasticity"] == "rehearsal"
        ]
        for key, style in [("items_m0_token", "-"), ("pairs_m0_token", "--")]:
            axes[0, 0].plot(
                ts,
                [np.mean([r[key] for r in a if r["step"] == t]) for t in ts],
                color=color,
                ls=style,
                label=f"{kind}: {'items' if 'items' in key else 'pairs'}",
            )
        if kind == "competitive":
            for key, style in [("drift_u_one_minus_cka", "-"), ("drift_c_one_minus_cka", "--")]:
                axes[0, 1].plot(
                    ts,
                    [np.mean([r[key] for r in a if r["step"] == t]) for t in ts],
                    color=color,
                    ls=style,
                    label="lower" if "_u_" in key else "higher",
                )
    axes[0, 0].set(
        title="Old memories under noise + rehearsal",
        xlabel="Postlearning step",
        ylabel="Clean token accuracy",
        ylim=(0.5, 1.03),
    )
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(
        title="Competitive representation geometry",
        xlabel="Postlearning step",
        ylabel="1 − linear CKA",
    )
    axes[0, 1].legend()
    lesions = ["feedback_off", "chunk_lesion", "lower_lesion"]
    a = [
        r
        for r in rows
        if r["kind"] == "competitive"
        and r["noise"] == 0.08
        and r["intervention"] == "intact"
        and r["plasticity"] == "rehearsal"
        and r["step"] == 100
    ]
    acute = [
        np.mean([r["robustness_token"] - r["acute"][k]["robustness_token"] for r in a])
        for k in lesions
    ]
    intact = np.mean([r["robustness_token"] for r in a])
    chronic = [
        intact
        - np.mean(
            [
                r["robustness_token"]
                for r in rows
                if r["kind"] == "competitive"
                and r["noise"] == 0.08
                and r["plasticity"] == "rehearsal"
                and r["step"] == 100
                and r["intervention"] == k
            ]
        )
        for k in lesions
    ]
    xx = np.arange(3)
    axes[1, 0].bar(xx - 0.17, acute, 0.34, label="Acute at final checkpoint", color="#a63a50")
    axes[1, 0].bar(
        xx + 0.17, chronic, 0.34, label="Chronic from postlearning start", color="#218380"
    )
    axes[1, 0].axhline(0.05, color="black", ls=":", label="Registered 5-point threshold")
    axes[1, 0].axhline(0, color="grey", lw=0.5)
    axes[1, 0].set(
        xticks=xx,
        xticklabels=["Feedback off", "Chunk lesion", "Lower lesion"],
        ylabel="Intact minus lesioned robustness",
        title="Feedback is weakly used on this task",
    )
    axes[1, 0].legend(fontsize=8)
    summary = json.loads((root / "summary.json").read_text())
    for i, p in enumerate(summary["primary"].values()):
        v = p["contrasts"]["robustness_token"]
        axes[1, 1].scatter(v["per_seed"], np.full(8, i), color="#a63a50", alpha=0.7)
        axes[1, 1].plot(v["ci95"], [i + 0.12, i + 0.12], color="black")
        axes[1, 1].scatter(v["mean"], i + 0.12, color="black", marker="D", s=25)
    axes[1, 1].axvline(0, color="grey", lw=0.7)
    axes[1, 1].axvline(0.05, color="black", ls=":")
    axes[1, 1].set(
        yticks=[0, 1],
        yticklabels=["Frozen", "Rehearsal"],
        xlabel="Competitive minus matched control",
        title="Robustness: seeds and paired 95% interval",
        ylim=(-0.4, 1.4),
    )
    fig.suptitle("Descriptive diagnostics after the registered run · no model retuning")
    fig.savefig(root / "diagnostics.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
