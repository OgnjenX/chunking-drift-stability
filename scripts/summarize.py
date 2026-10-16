"""Produce aggregate data and figures directly from archived observations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srclearner.metrics import paired_interval


def main():
    p = Path("results/development-v2")
    rows = json.loads((p / "metrics.json").read_text())
    modes = [
        "full",
        "frozen",
        "no_chunk",
        "no_feedback",
        "no_predictive",
        "no_competition",
        "ngram",
    ]
    agg = []
    for noise in [0, 0.1, 0.25]:
        full = [r for r in rows if r["mode"] == "full" and r["noise"] == noise]
        for mode in modes:
            selected = [r for r in rows if r["mode"] == mode and r["noise"] == noise]
            agg.append(
                dict(
                    mode=mode,
                    noise=noise,
                    mean_accuracy=float(np.mean([r["accuracy"] for r in selected])),
                    full_minus_control=paired_interval(
                        [r["accuracy"] for r in full], [r["accuracy"] for r in selected]
                    ),
                )
            )
    (p / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), layout="constrained")
    for mode in modes:
        values = [r["mean_accuracy"] for r in agg if r["mode"] == mode]
        axes[0].plot([0, 0.1, 0.25], values, marker="o", label=mode.replace("_", " "))
    axes[0].set(
        ylim=(0, 1.04),
        xlabel="Probability of replacing each input token",
        ylabel="Mean accuracy",
        title="Context transfer under sensory corruption",
    )
    axes[0].axhline(0.25, color="gray", ls=":")
    axes[0].legend(fontsize=8)
    for i, mode in enumerate(modes):
        vals = [r["accuracy"] for r in rows if r["mode"] == mode and r["noise"] == 0.1]
        axes[1].bar(i, np.mean(vals), color="#356a91" if mode == "full" else "#a8b8c1")
        axes[1].scatter([i] * 3, vals, c="black", s=22, zorder=3)
    axes[1].set(
        xticks=range(len(modes)),
        xticklabels=[m.replace("_", "\n") for m in modes],
        ylim=(0, 1.04),
        ylabel="Accuracy",
        title="Prespecified 0.10 corruption: each world shown",
    )
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].axhline(0.25, color="gray", ls=":")
    fig.suptitle("Exploratory follow-up, fresh worlds 311–313; no confirmatory claim")
    fig.savefig(p / "headroom.png", dpi=170)
    plt.close(fig)
    print("Follow-up means:")
    for r in agg:
        print(r["mode"], r["noise"], round(r["mean_accuracy"], 4))
    print("Full old/tags", [r for r in rows if r["mode"] == "full" and r["noise"] == 0])


if __name__ == "__main__":
    main()
