"""Numerical/refinement and novel-composition diagnostics for the unit variant."""

import json
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np

from ..controls import Readout, scores
from ..develop_probe import DEV_LONG, DEV_OLD
from ..model import Params
from ..task import LONG, stream
from .model import PhysicalUnitsField


def assess_variant(results, mechanism, numeric):
    # Descriptive post-development screens, not retroactive confirmatory tests.
    """Apply descriptive readiness screens without treating them as confirmation."""
    coverage = all(m["trained_winners"] / 20 >= 0.75 for m in mechanism)
    winner_gains = [(m["trained_winners"] - m["untrained_filter_winners"]) / 20 for m in mechanism]
    performance_gains = {
        r["condition"]: r["field"] - r["frozen_filters"] for r in results["performance"]
    }
    learning_benefit = max(performance_gains.values()) >= 0.05
    emergence = coverage and min(winner_gains) >= 0.1 and learning_benefit
    integration = max(r["maximum_relative_state_error"] for r in numeric) < 0.02
    eligible = (
        coverage
        and emergence
        and learning_benefit
        and integration
        and results["shared_headroom_calibrated"]
    )
    readiness = {
        "status": "eligible_for_further_Stage1_validation"
        if eligible
        else "STOPPED_AT_DEVELOPMENT",
        "variant": "source-unit conversion",
        "selective_winner_coverage": coverage,
        "learned_emergence_screen_passed": emergence,
        "filter_learning_benefit_screen_passed": learning_benefit,
        "winner_coverage_gains": winner_gains,
        "learned_minus_frozen_exact_by_condition": performance_gains,
        "shared_headroom_calibrated": results["shared_headroom_calibrated"],
        "integration_refinement": integration,
        "maximum_numerical_refinement_error": max(
            r["maximum_relative_state_error"] for r in numeric
        ),
        "confirmation_seeds_executed": [],
        "stage1_confirmation": "not_run",
        "stage2": "not_run",
        "stage3": "not_run",
        "stability_hypothesis_verdict": "NOT_TESTED",
        "reason": (
            "Only a passing readiness screen can permit further Stage 1 validation; "
            "no confirmatory verdict is produced."
        ),
    }
    return readiness


def main():
    """Measure numerical refinement and novel sequence performance."""
    root = Path("exp002/results/unit-consistency-calibration")
    p = Params(**json.loads((root / "decision.json").read_text())["parameters"])
    numeric = []
    novel = []
    sequences = [
        "".join(t)
        for k in (3, 4)
        for t in permutations("ABCD", k)
        if "".join(t) not in DEV_LONG + LONG
    ][:16]
    for seed in [101, 102, 103]:
        z = np.load(root / f"s{seed}_codes.npz")
        m = PhysicalUnitsField(seed, p)
        m.W = z["W"].copy()
        m.M = z["M"].copy()
        h = Readout(len(m.state()))
        h.B = z["head"].copy()
        frozen = PhysicalUnitsField(seed, p)
        fh = Readout(len(m.state()))
        fh.B = z["frozen_head"].copy()
        for dt in [0.025, 0.0125]:
            fine = PhysicalUnitsField(seed, replace(p, dt=dt))
            fine.W = m.W.copy()
            fine.M = m.M.copy()
            differences = []
            for seq in DEV_OLD + DEV_LONG:
                a = m.run(stream(seq, p, np.random.default_rng(0), delay=0.5))[-1]
                b = fine.run(stream(seq, fine.p, np.random.default_rng(0), delay=0.5))[-1]
                differences.append(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))
            numeric.append(
                {"seed": seed, "dt": dt, "maximum_relative_state_error": float(max(differences))}
            )
        for rep in range(4):
            for i, seq in enumerate(sequences):
                a = stream(
                    seq, p, np.random.default_rng([seed, 963, rep, i]), delay=2.0, noise=0.005
                )
                for label, model, head in [("field", m, h), ("frozen_filters", frozen, fh)]:
                    s = model.run(a)[-1]
                    novel.append(
                        {
                            "seed": seed,
                            "rep": rep,
                            "sequence": seq,
                            "model": label,
                            **scores(head.predict(s), seq),
                        }
                    )
    (root / "numerics.json").write_text(json.dumps(numeric, indent=2) + "\n")
    (root / "novel-compositions.json").write_text(json.dumps(novel, indent=2) + "\n")
    results = json.loads((root / "decision.json").read_text())
    mechanism = json.loads((root / "mechanisms.json").read_text())
    readiness = assess_variant(results, mechanism, numeric)
    (root / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
    print(json.dumps(readiness, indent=2))
    print(
        {
            label: float(np.mean([r["exact"] for r in novel if r["model"] == label]))
            for label in ["field", "frozen_filters"]
        }
    )


if __name__ == "__main__":
    main()
