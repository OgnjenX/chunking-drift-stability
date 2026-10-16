"""Independent development/calibration, separate from frozen confirmation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .develop_probe import DEV_LONG, DEV_OLD
from .evaluate import evaluate, fit, serialize, summary
from .model import Params, TemporalField
from .task import stream

LEVELS = [
    {"name": "clean", "delay": 0.5, "noise": 0.0, "missing": 0, "interference": 0.0},
    {"name": "light", "delay": 2.0, "noise": 0.005, "missing": 0, "interference": 0.1},
    {"name": "medium", "delay": 4.0, "noise": 0.01, "missing": 0, "interference": 0.2},
    {"name": "partial", "delay": 6.0, "noise": 0.02, "missing": 1, "interference": 0.2},
    {"name": "hard", "delay": 10.0, "noise": 0.03, "missing": 1, "interference": 0.4},
]


def main():
    """Evaluate the predefined development difficulty ladder across controls."""
    out = Path("exp002/results/development-calibration")
    out.mkdir(exist_ok=False, parents=True)
    # Source gain 8 and 2014 pulse duration .75 retained; not selected for advantage.
    p = Params(pulse=0.75, feedback=8.0)
    all_rows = []
    train_records = []
    numerics = []
    for seed in [101, 102, 103]:
        models, phases, initial = fit(seed, p, DEV_OLD, DEV_LONG, repetitions=30)
        for phase in phases:
            train_records.append({"seed": seed, **phase})
        f = models["field"]
        # Compare same trained filters under successively finer clean integration.
        for dt in [0.025, 0.0125]:
            fine = TemporalField(seed, replace(p, dt=dt))
            fine.W = f.W.copy()
            fine.M = f.M.copy()
            errors = []
            for seq in DEV_LONG:
                a = f.run(stream(seq, p, np.random.default_rng(0), delay=0.5))[-1]
                b = fine.run(stream(seq, fine.p, np.random.default_rng(0), delay=0.5))[-1]
                errors.append(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
            numerics.append(
                {
                    "seed": seed,
                    "base_dt": p.dt,
                    "fine_dt": dt,
                    "max_relative_state_difference": max(errors),
                }
            )
        for level in LEVELS:
            difficulty = {k: v for k, v in level.items() if k != "name"}
            rows = evaluate(models, DEV_OLD + DEV_LONG, seed, difficulty, repeats=4, phase=91)
            for row in rows:
                row["difficulty"] = level["name"]
                row.pop("state", None)
            all_rows.extend(rows)
            print(seed, level["name"], summary(rows), flush=True)
        f.save(out / f"s{seed}_field.npz")
        np.savez_compressed(
            out / f"s{seed}_readouts.npz",
            field=models["head"].B,
            wm=models["no_head"].B,
            initial_W=initial,
            **models["gru"].w,
        )
        for name, data in [
            ("trials", all_rows),
            ("training", train_records),
            ("numerics", numerics),
        ]:
            (out / f"{name}.json").write_text(json.dumps(data, default=serialize, indent=2) + "\n")
    calibration = []
    for level in LEVELS[1:]:
        sub = [r for r in all_rows if r["difficulty"] == level["name"]]
        result = summary(sub)
        ok = all(0.45 <= result[m]["exact"] <= 0.85 for m in ("field", "gru"))
        distance = sum(abs(result[m]["exact"] - 0.65) for m in ("field", "gru"))
        calibration.append(
            {
                "difficulty": level,
                "performance": result,
                "eligible": ok,
                "distance_from_target": distance,
            }
        )
    eligible = [x for x in calibration if x["eligible"]]
    selected = min(eligible, key=lambda x: x["distance_from_target"]) if eligible else None
    decision = {
        "parameters": asdict(p),
        "seeds": [101, 102, 103],
        "calibration": calibration,
        "selected": selected,
        "headroom_calibrated": bool(eligible),
        "numerical_max_difference": max(r["max_relative_state_difference"] for r in numerics),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("Development calibration done.", flush=True)


if __name__ == "__main__":
    main()
