"""Isolated source-law validation; not a full source-paper reproduction."""

import argparse
import json
from pathlib import Path

import numpy as np

from .mechanisms import instar, working_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/reference-v1")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("Refusing overwrite")
    out.mkdir(parents=True)
    records = []
    for sequence in [[0, 1], [1, 0], [0, 1, 2], [2, 1, 0]]:
        x = working_memory(sequence)
        w = np.ones(14) / 14
        initial = w.copy()
        for step in range(31):
            analytic = x + (initial - x) * np.exp(-0.2 * step)
            records.append(
                dict(
                    sequence=sequence,
                    step=step,
                    weights=w.tolist(),
                    target=x.tolist(),
                    error=float(np.linalg.norm(w - x)),
                    analytic_error=float(np.max(np.abs(w - analytic))),
                )
            )
            w = instar(w, x)
    (out / "trajectories.json").write_text(json.dumps(records, indent=2) + "\n")
    summary = dict(
        max_analytic_error=max(r["analytic_error"] for r in records),
        start_error=float(np.mean([r["error"] for r in records if r["step"] == 0])),
        end_error=float(np.mean([r["error"] for r in records if r["step"] == 30])),
        scope="Isolated constant-input instar only; not coupled Masking Field",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for seq in [[0, 1], [1, 0], [0, 1, 2], [2, 1, 0]]:
        rs = [r for r in records if r["sequence"] == seq]
        ax.plot([r["step"] for r in rs], [r["error"] for r in rs], label=str(seq))
    ax.axhline(
        float(summary["start_error"]), ls=":", color="gray", label="Frozen mean initial error"
    )
    ax.set(
        xlabel="Local updates",
        ylabel="Distance to normalized temporal memory",
        title="Source-law check: learned filter approaches its observed input",
    )
    ax.legend()
    fig.savefig(out / "tuning.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
