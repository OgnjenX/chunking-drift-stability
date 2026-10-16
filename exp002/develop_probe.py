"""First bounded exploratory source-circuit characterization, no confirmation."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import Params, TemporalField
from .task import stream

DEV_OLD = ["A", "B", "C", "D", "AC", "CA", "AD", "DA", "BD", "DB"]
DEV_LONG = ["ACD", "ADC", "CAD", "CBD", "ACDB", "ADCB", "ADBC", "CADB", "BDAC", "DBCA"]


def main():
    """Explore source-circuit selectivity using development sequences only."""
    out = Path("exp002/results/development-source-probe")
    out.mkdir(exist_ok=False, parents=True)
    rows = []
    for pulse in [0.75, 2.0, 6.0]:
        for feedback in [1.0, 8.0]:
            p = Params(pulse=pulse, feedback=feedback)
            m = TemporalField(101, p)
            before = m.W.copy()
            snapshots = []
            for rep in range(8):
                for i, seq in enumerate(DEV_OLD + DEV_LONG):
                    a = stream(seq, p, np.random.default_rng([101, rep, i]), delay=2.0)
                    m.run(a, learn=True)
            weights = float(np.linalg.norm(m.W - before) / np.linalg.norm(before))
            for seq in DEV_OLD + DEV_LONG:
                a = stream(seq, p, np.random.default_rng(103), delay=2.0)
                t = m.run(a)
                snapshots.append(t[-1])
            s = np.array(snapshots)
            c = s[:, 8:]
            row = {
                "pulse": pulse,
                "feedback": feedback,
                "parameters": asdict(p),
                "weight_relative_change": weights,
                "occupied_winners": len(set(c.argmax(axis=1))),
                "mean_final_x": float(s[:, :4].mean()),
                "mean_final_y": float(s[:, 4:8].mean()),
                "max_c": float(c.max()),
                "c_variation": float(c.std(axis=0).mean()),
            }
            rows.append(row)
            np.savez_compressed(
                out / f"p{pulse}_f{feedback}.npz", states=s, W=m.W, M=m.M, before=before
            )
            (out / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
