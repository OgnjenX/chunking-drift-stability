"""Post-run audit: independently reconstruct saved states and replay one branch.

Added after the registered experiment; does not alter its protocol or analysis.
"""
import json
from pathlib import Path
import numpy as np
from .data import encode, LEARNED
from .model import Model
from .run import train


def main():
    root = Path("results/registered-v1")
    config = json.loads((root/"config.json").read_text())
    rows = json.loads((root/"metrics.json").read_text())
    assert len(rows) == 600
    x, target = encode(LEARNED)
    valid = target != 8
    max_error = 0.0
    for row in rows:
        seed, kind = row["seed"], row["kind"]
        label = f"s{seed}_{kind}_n{row['noise']}_{row['plasticity']}_{row['intervention']}_t{row['step']}"
        z = np.load(root/"states"/f"{label}.npz")
        u = np.tanh(x @ z["E"]) * z["lower_mask"]
        if kind == "flat":
            c = np.zeros((len(x), 0))
            h = u
        else:
            scores = u @ z["P"].T / config["temperature"]
            if kind == "competitive":
                es = np.exp(scores-scores.max(axis=1, keepdims=True))
                c = es/es.sum(axis=1, keepdims=True)
            else:
                c = np.tanh(scores)/np.sqrt(config["chunks"])
            c *= z["chunk_mask"]
            h = (float(z["lower_gain"])*u + float(z["feedback_gain"])*(c @ z["B"])) * z["lower_mask"]
        for key, a in [("u", u), ("c", c), ("h", h)]:
            np.testing.assert_allclose(a, z[key], rtol=1e-12, atol=1e-12)
            if a.size:
                max_error = max(max_error, float(np.abs(a-z[key]).max()))
        pred = (h @ z["D"]).reshape(-1,4,9).argmax(axis=2)
        score = float((pred == target)[valid].mean())
        assert abs(score-row["learned_m0_token"]) < 1e-14
        exact = float(np.all((pred == target) | ~valid, axis=1).mean())
        assert abs(exact-row["learned_m0_exact"]) < 1e-14
        if row["noise"] == 0 and row["plasticity"] == "frozen":
            b = np.load(root/"states"/f"s{seed}_{kind}_phase2.npz")
            for name in ("E", "D", "u", "c", "h"):
                np.testing.assert_array_equal(z[name], b[name])
    # Reconstruct a trained checkpoint and repeat the full stochastic trajectory.
    seed, kind = 0, "competitive"
    model = Model(kind, seed, config["hidden"], config["chunks"], config["feedback_gain"], config["temperature"])
    b = np.load(root/"states"/f"s{seed}_{kind}_phase2.npz")
    model.w = {k: b[k].copy() for k in model.w}
    scales = {k: float(np.sqrt(np.mean(v*v))) for k,v in model.w.items()}
    for step in range(1, config["post_steps"]+1):
        model.noise(seed, step, .08, scales)
        train(model, LEARNED, config["post_rehearsal_epochs"], config, seed, 4,
              (step-1)*config["post_rehearsal_epochs"])
    final = np.load(root/"states"/"s0_competitive_n0.08_rehearsal_intact_t100.npz")
    for key in model.w:
        np.testing.assert_array_equal(model.w[key], final[key])
    result = {"saved_states_independently_reconstructed": len(rows),
              "clean_token_and_exact_scores_verified": len(rows),
              "zero_noise_frozen_branches_identical": True,
              "seed0_rehearsal_trajectory_replayed_bitwise": True,
              "max_saved_response_absolute_error": max_error}
    (root/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
