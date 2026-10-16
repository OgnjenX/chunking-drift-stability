"""Run the registered experiment once, refusing to overwrite an output directory."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import HELDOUT, ITEMS, LEARNED, NEW, OLD, PAIRS, corrupt, encode
from .metrics import change, code_stats, probe, structure_contrast
from .model import Model


def train(model, sequences, epochs, config, seed, phase, start=0):
    """Apply the specified seeded training budget and return epoch losses."""
    losses = []
    for epoch in range(start, start + epochs):
        rng = np.random.default_rng([seed, phase, epoch, 401])
        x, target = corrupt(sequences, rng, probability=config["training_mask_probability"])
        losses.append(model.update(x, target, config["learning_rate"]))
    return losses


def evaluate(model, config, baseline=None, anchor=None):
    """Measure function, probe robustness, and change relative to saved anchors."""
    metrics = {}
    for name, seqs in [
        ("items", ITEMS),
        ("pairs", PAIRS),
        ("new", NEW),
        ("learned", LEARNED),
        ("heldout", HELDOUT),
    ]:
        for missing in (0, 1):
            result = probe(model, seqs, missing, config["probe_repeats"])
            for kind, value in result.items():
                metrics[f"{name}_m{missing}_{kind}"] = value
    levels = config["robustness_mask_levels"]
    for kind in ("token", "exact"):
        metrics[f"robustness_{kind}"] = float(
            np.mean([probe(model, NEW, m, config["probe_repeats"])[kind] for m in levels])
        )
    if anchor is not None:
        metrics["anchored_learned_token"] = probe(
            model, LEARNED, 0, config["probe_repeats"], anchor
        )["token"]
    a = model.forward(encode(LEARNED)[0])
    for key in ("u", "c", "h"):
        if baseline is not None:
            for metric, value in change(baseline[key], a[key]).items():
                metrics[f"drift_{key}_{metric}"] = value
    metrics.update(code_stats(a["c"]))
    if baseline is not None and a["c"].shape[1]:
        metrics["winner_agreement"] = float(
            np.mean(baseline["c"].argmax(axis=1) == a["c"].argmax(axis=1))
        )
    return metrics, a


def state(path, model, arrays):
    """Save weights, intervention settings, and representation probes."""
    np.savez_compressed(
        path,
        **model.w,
        lower_mask=model.lower_mask,
        chunk_mask=model.chunk_mask,
        feedback_gain=model.gain,
        lower_gain=model.lower_gain,
        **{k: arrays[k] for k in ("u", "c", "h")},
    )


def digest(path):
    """Hash the exact bytes of a configuration or source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Execute the registered schedule in a new output directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered.json")
    parser.add_argument("--output", default="results/registered-v1")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "states").mkdir()
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    metadata = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "commit": git("rev-parse", "HEAD"),
        "git_status_before_run": git("status", "--porcelain"),
        "config_sha256": digest(args.config),
        "command": sys.argv,
        "source_sha256": {str(p): digest(p) for p in sorted(Path("chunkdrift").glob("*.py"))},
        "threads": {
            k: os.getenv(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows, learning, composition, losses = [], [], [], []
    for seed in config["seeds"]:
        for kind in ("flat", "dense", "competitive"):
            model = Model(
                kind,
                seed,
                config["hidden"],
                config["chunks"],
                config["feedback_gain"],
                config["temperature"],
            )
            old_a = {}
            for phase, seqs, epochs in [
                (1, OLD, config["epochs_old"]),
                (2, NEW, config["epochs_new"]),
            ]:
                trace = train(model, seqs, epochs, config, seed, phase)
                losses.append({"seed": seed, "kind": kind, "phase": phase, "loss": trace})
                metrics, arrays = evaluate(model, config)
                learning.append(
                    {
                        "seed": seed,
                        "kind": kind,
                        "phase": phase,
                        "parameters": model.parameter_count,
                        **metrics,
                    }
                )
                state(out / "states" / f"s{seed}_{kind}_phase{phase}.npz", model, arrays)
                old_arrays = model.forward(encode(OLD)[0])
                if phase == 1:
                    old_a = {k: old_arrays[k].copy() for k in ("u", "c", "h")}
                else:
                    learning[-1]["recoding"] = {
                        k: change(previous, old_arrays[k]) for k, previous in old_a.items()
                    }
                    learning[-1]["recoding_structure_contrast"] = structure_contrast(
                        old_a["u"], old_arrays["u"], OLD
                    )
            baseline = model.forward(encode(LEARNED)[0])
            scales = {k: float(np.sqrt(np.mean(v * v))) for k, v in model.w.items()}
            anchor = model.w["D"].copy()
            # A separate branch tests reuse; it never changes the drift baseline.
            branch = model.copy()
            previous = 0
            for budget in config["composition_epochs"]:
                train(branch, HELDOUT, budget - previous, config, seed, 3, previous)
                metrics, _ = evaluate(branch, config)
                composition.append({"seed": seed, "kind": kind, "budget": budget, **metrics})
                previous = budget
            conditions = [
                (noise, plasticity, "intact")
                for noise in config["noise_levels"]
                for plasticity in ("frozen", "rehearsal")
            ]
            if kind == "competitive":
                conditions.extend(
                    (max(config["noise_levels"]), "rehearsal", intervention)
                    for intervention in ("feedback_off", "chunk_lesion", "lower_lesion")
                )
            for noise, plasticity, intervention in conditions:
                branch = model.copy()
                if intervention != "intact":
                    branch.intervene(intervention, seed, config["disruption_fraction"])
                label = f"s{seed}_{kind}_n{noise}_{plasticity}_{intervention}"
                for step in range(config["post_steps"] + 1):
                    if step:
                        branch.noise(seed, step, noise, scales)
                        if plasticity == "rehearsal":
                            train(
                                branch,
                                LEARNED,
                                config["post_rehearsal_epochs"],
                                config,
                                seed,
                                4,
                                (step - 1) * config["post_rehearsal_epochs"],
                            )
                    if step in config["checkpoints"]:
                        metrics, arrays = evaluate(branch, config, baseline, anchor)
                        row = {
                            "seed": seed,
                            "kind": kind,
                            "noise": noise,
                            "plasticity": plasticity,
                            "intervention": intervention,
                            "step": step,
                            **metrics,
                        }
                        if kind == "competitive" and intervention == "intact":
                            row["acute"] = {}
                            for lesion in ("feedback_off", "chunk_lesion", "lower_lesion"):
                                acute = branch.copy()
                                acute.intervene(lesion, seed, config["disruption_fraction"])
                                row["acute"][lesion] = evaluate(acute, config)[0]
                        all_rows.append(row)
                        state(out / "states" / f"{label}_t{step}.npz", branch, arrays)
            print(f"Finished seed={seed} kind={kind}", flush=True)
            # Save progressive evidence even if a later seed fails.
            for name, data in [
                ("metrics", all_rows),
                ("learning", learning),
                ("composition", composition),
                ("losses", losses),
            ]:
                (out / f"{name}.json").write_text(
                    json.dumps(data, indent=2, allow_nan=False) + "\n"
                )
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
