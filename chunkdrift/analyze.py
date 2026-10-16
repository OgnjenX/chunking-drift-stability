"""Frozen decision rules plus descriptive plots; no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np


def paired(a, b, samples=10000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(20261016)
    boot = d[rng.integers(0, len(d), (samples, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "per_seed": d.tolist()}


def compare(rows, left, right, metric, seeds, samples):
    def values(select):
        subset = [r for r in rows if all(r[k] == v for k, v in select.items())]
        by_seed = {r["seed"]: r[metric] for r in subset}
        assert len(subset) == len(seeds) == len(by_seed), (select, metric)
        return [by_seed[s] for s in seeds]
    a, b = values(left), values(right)
    return {**paired(a, b, samples), "left_mean": float(np.mean(a)), "right_mean": float(np.mean(b))}


def analyze(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    rows = json.loads((root / "metrics.json").read_text())
    learning = json.loads((root / "learning.json").read_text())
    composition = json.loads((root / "composition.json").read_text())
    seeds, samples = config["seeds"], config["bootstrap_samples"]
    n, end, effect = max(config["noise_levels"]), config["post_steps"], config["meaningful_effect"]
    def mean(subset, key):
        return float(np.mean([r[key] for r in subset]))
    acquisition = {}
    for kind in ("flat", "dense", "competitive"):
        a = [r for r in learning if r["kind"] == kind and r["phase"] == 1]
        b = [r for r in learning if r["kind"] == kind and r["phase"] == 2]
        # OLD consists of 8 singletons and 4 pairs: equal total item counts per group.
        old = [(r["items_m0_token"] + r["pairs_m0_token"])/2 for r in a]
        acquired = float(np.mean(old)) >= config["acquisition_gate"] and mean(b, "learned_m0_token") >= config["acquisition_gate"]
        acquisition[kind] = {"old_after_A": float(np.mean(old)), "learned_after_B": mean(b, "learned_m0_token"),
                             "passed": acquired, "old_after_A_per_seed": old,
                             "learned_after_B_per_seed": [r["learned_m0_token"] for r in b],
                             "parameters": b[0]["parameters"]}
    gate = acquisition["dense"]["passed"] and acquisition["competitive"]["passed"]
    primary = {}
    all_chronic = {}
    for schedule in ("frozen", "rehearsal"):
        shared = {"noise": n, "step": end, "plasticity": schedule, "intervention": "intact"}
        left, right = {**shared, "kind": "competitive"}, {**shared, "kind": "dense"}
        contrasts = {m: compare(rows, left, right, m, seeds, samples)
                     for m in ("learned_m0_token", "robustness_token", "new_m1_masked_token")}
        selected = [r for r in rows if all(r[k] == v for k, v in left.items())]
        means = {m: mean(selected, m) for m in ("items_m0_token", "pairs_m0_token", "drift_u_cosine",
                 "drift_u_relative_rms", "drift_c_cosine", "learned_m0_token")}
        advantage = all(contrasts[m]["mean"] >= effect and contrasts[m]["ci95"][0] > 0
                        for m in ("learned_m0_token", "robustness_token"))
        retention = means["learned_m0_token"] >= config["retention_gate"] and min(means["items_m0_token"], means["pairs_m0_token"]) >= config["retention_gate"]
        drift = means["drift_u_cosine"] >= config["minimum_drift_cosine"] and means["drift_u_relative_rms"] >= config["minimum_drift_relative_rms"]
        stability = means["drift_c_cosine"] <= config["hierarchy_stability_ratio"] * means["drift_u_cosine"]
        baseline_c = [r for r in learning if r["kind"] == "competitive" and r["phase"] == 2]
        not_collapsed = mean(baseline_c, "winner_count") > 1
        if schedule == "rehearsal":
            for lesion in ("feedback_off", "chunk_lesion", "lower_lesion"):
                all_chronic[lesion] = compare(rows, left, {**left, "intervention": lesion}, "robustness_token", seeds, samples)
        lesion_gate = schedule == "rehearsal" and all(all_chronic[k]["mean"] >= effect and all_chronic[k]["ci95"][0] > 0
                                                       for k in ("feedback_off", "chunk_lesion"))
        # Frozen chronic ablations were not run: no full-mechanism support there.
        primary[schedule] = {"contrasts": contrasts, "competitive_means": means,
                             "acquisition_gate": gate, "advantage_gate": advantage,
                             "retention_gate": retention, "drift_gate": drift,
                             "higher_stability_gate": stability, "noncollapsed_gate": not_collapsed,
                             "chronic_lesion_gate": lesion_gate,
                             "conditional_robustness_support": bool(gate and advantage and retention),
                             "full_mechanism_support": bool(gate and advantage and retention and drift and stability and not_collapsed and lesion_gate),
                             "ceiling_at_B": min(acquisition[k]["learned_after_B"] for k in ("dense", "competitive")) >= .98,
                             "final_retention_ceiling": min(contrasts["learned_m0_token"][k] for k in ("left_mean", "right_mean")) >= .98}
    # Collapse repeated budgets to one seed-level statistic before bootstrapping.
    reuse = []
    for seed in seeds:
        for kind in ("dense", "competitive", "flat"):
            selected = [r for r in composition if r["seed"] == seed and r["kind"] == kind]
            reuse.append({"seed": seed, "kind": kind,
                          "score": mean(selected, "heldout_m1_masked_token")})
    reuse_result = compare(reuse, {"kind": "competitive"}, {"kind": "dense"}, "score", seeds, samples)
    final_comp = [r for r in composition if r["kind"] == "competitive" and r["budget"] == max(config["composition_epochs"])]
    old_mean = (mean(final_comp, "items_m0_token") + mean(final_comp, "pairs_m0_token"))/2
    reuse_result.update({"old_retention": old_mean,
                         "supported": bool(gate and reuse_result["mean"] >= effect and reuse_result["ci95"][0] > 0 and old_mean >= config["retention_gate"])})
    recoding = {}
    for kind in ("flat", "dense", "competitive"):
        selected = [r for r in learning if r["kind"] == kind and r["phase"] == 2]
        recoding[kind] = {"lower_cosine": float(np.mean([r["recoding"]["u"]["cosine"] for r in selected])),
                          "structure_contrast": mean(selected, "recoding_structure_contrast"),
                          "old_items": mean(selected, "items_m0_token"), "old_pairs": mean(selected, "pairs_m0_token"),
                          "winner_count": mean(selected, "winner_count") if kind != "flat" else None,
                          "entropy": mean(selected, "normalized_entropy") if kind != "flat" else None}
    summary = {"acquisition": acquisition, "primary": primary, "chronic_lesions": all_chronic,
               "reuse": reuse_result, "recoding": recoding,
               "row_counts": {"metrics": len(rows), "learning": len(learning), "composition": len(composition)}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    table = ["# Registered results", "", "Generated from saved per-seed measurements. Interpret with docs/protocol.md.", "",
             "| Model | Parameters | A old clean | B all clean | B old items | B old pairs | Recoding cosine |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kind, a in acquisition.items():
        r = recoding[kind]
        table.append(f"| {kind} | {a['parameters']} | {a['old_after_A']:.3f} | {a['learned_after_B']:.3f} | {r['old_items']:.3f} | {r['old_pairs']:.3f} | {r['lower_cosine']:.3f} |")
    table += ["", "Competitive minus capacity-matched dense, step 100/noise 0.08:", "",
              "| Schedule | Clean retention difference [95% CI] | Robustness difference [95% CI] | Lower/higher cosine drift | Full gates passed |",
              "|---|---:|---:|---:|---|"]
    for name, p in primary.items():
        f = lambda x: f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
        table.append(f"| {name} | {f(p['contrasts']['learned_m0_token'])} | {f(p['contrasts']['robustness_token'])} | {p['competitive_means']['drift_u_cosine']:.3f} / {p['competitive_means']['drift_c_cosine']:.3f} | {p['full_mechanism_support']} |")
    table += ["", f"Reuse masked-target difference: {reuse_result['mean']:+.3f}, 95% CI {reuse_result['ci95']}. Gate passed: {reuse_result['supported']}.",
              "", "All intervals resample the eight paired seeds. They do not cover task or hyperparameter uncertainty.",
              "Acquisition failures, per-seed values, lesion gates, and ceilings are in summary.json."]
    (root / "tables.md").write_text("\n".join(table) + "\n")
    plot(root, rows, composition, config)
    return summary


def plot(root, rows, composition, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"flat": "#777777", "dense": "#218380", "competitive": "#a63a50"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    n = max(config["noise_levels"])
    for col, schedule in enumerate(("frozen", "rehearsal")):
        for kind, color in colors.items():
            subset = [r for r in rows if r["noise"] == n and r["plasticity"] == schedule and r["kind"] == kind and r["intervention"] == "intact"]
            ts = config["checkpoints"]
            for ax, metric in [(axes[0, col], "learned_m0_token"), (axes[1, col], "robustness_token")]:
                for seed in config["seeds"]:
                    ax.plot(ts, [next(r[metric] for r in subset if r["seed"] == seed and r["step"] == t) for t in ts], color=color, alpha=.17, lw=.7)
                ax.plot(ts, [np.mean([r[metric] for r in subset if r["step"] == t]) for t in ts], color=color, label=kind)
                ax.set(xlabel="Postlearning step", ylim=(0, 1.03))
        axes[0, col].set_title(f"Clean retention: {schedule}")
        axes[1, col].set_title(f"Missing-item robustness: {schedule}")
        axes[0, col].legend(fontsize=8)
    for kind, color in colors.items():
        subset = [r for r in rows if r["noise"] == n and r["plasticity"] == "rehearsal" and r["kind"] == kind and r["intervention"] == "intact"]
        axes[0, 2].plot(config["checkpoints"], [np.mean([r["drift_u_cosine"] for r in subset if r["step"] == t]) for t in config["checkpoints"]], color=color, label=f"{kind} lower")
        if kind == "competitive":
            axes[0, 2].plot(config["checkpoints"], [np.mean([r["drift_c_cosine"] for r in subset if r["step"] == t]) for t in config["checkpoints"]], color=color, ls="--", label="competitive higher")
        axes[1, 2].plot(config["composition_epochs"], [np.mean([r["heldout_m1_masked_token"] for r in composition if r["kind"] == kind and r["budget"] == b]) for b in config["composition_epochs"]], color=color, marker="o", label=kind)
    axes[0, 2].set(title="Coordinate drift under rehearsal", xlabel="Postlearning step", ylabel="Mean 1 − cosine")
    axes[0, 2].legend(fontsize=8)
    axes[1, 2].set(title="Held-out compositional reuse", xlabel="Few-shot training epochs", ylabel="Masked-item accuracy", ylim=(0, 1.03))
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Registered toy benchmark · eight seeds · fixed noise SD 0.08 × baseline weight RMS")
    fig.savefig(root / "overview.png", dpi=170)
    fig.savefig(root / "overview.pdf")
    plt.close(fig)
    # Directly inspect learned competitive codes rather than assigning categories.
    codes = np.load(root / "states" / "s0_competitive_phase2.npz")
    from .data import LEARNED
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(codes["c"], aspect="auto", cmap="magma", vmin=0)
    ax.set(yticks=np.arange(len(LEARNED)), yticklabels=LEARNED, xlabel="Competitive unit", title="Seed 0: learned clean-probe competitive activities")
    fig.colorbar(im, ax=ax, label="Softmax activity")
    fig.savefig(root / "competitive-codes.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result", nargs="?", default="results/registered-v1")
    args = p.parse_args()
    result = analyze(args.result)
    print(json.dumps({"primary": result["primary"], "reuse": result["reuse"]}, indent=2))


if __name__ == "__main__":
    main()
