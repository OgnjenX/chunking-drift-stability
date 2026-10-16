import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = Path("results/development-v1")
    rows = json.loads((p / "metrics.json").read_text())
    d = json.loads((p / "decision.json").read_text())
    modes = ["full", "frozen", "no_chunk", "no_feedback", "no_predictive", "no_competition"]
    selected = [r for r in rows if r["setting"] == d["selected_vigilance"] and r["mode"] in modes]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for k, key in enumerate(["novel", "old"]):
        for i, m in enumerate(modes):
            vals = [r[key] for r in selected if r["mode"] == m]
            ax[k].bar(i, np.mean(vals), color="#356a91" if m == "full" else "#a8b8c1")
            ax[k].scatter([i] * len(vals), vals, c="black", s=22, zorder=3)
        ax[k].axhline(0.25, color="#b44c43", ls=":", label="4-way chance")
        ax[k].set(
            xticks=range(len(modes)),
            xticklabels=[m.replace("_", "\n") for m in modes],
            ylim=(0, 1.05),
            ylabel="Accuracy",
            title="Unseen filler combinations" if k == 0 else "Fixed old examples",
        )
        ax[k].legend(loc="upper right", fontsize=8)
    fig.suptitle("Development only: each point is an independent world (n=3)")
    fig.savefig(p / "accuracy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
