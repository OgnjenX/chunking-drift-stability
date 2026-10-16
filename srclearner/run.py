"""Development only. Confirmatory execution deliberately has no automatic path."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .controls import NgramDelta, learned_oracle
from .domain import World
from .mechanisms import Learner
from .metrics import accuracy, paired_interval


def source_manifest():
    files = sorted(
        list(Path("srclearner").glob("*.py"))
        + list(Path("tests").glob("*.py"))
        + list(Path("configs").glob("*.json"))
        + list(Path("docs").glob("*.md"))
        + [Path("pyproject.toml")]
    )
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/development.json")
    ap.add_argument("--output", default="results/development-v1")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("Refusing to overwrite an attempt. Choose a new output directory.")
    cfg = json.loads(Path(args.config).read_text())
    if set(cfg["seeds"]) & set(cfg["reserved_confirmation_seeds"]):
        raise SystemExit("Development must not consume reserved confirmation seeds.")
    out.mkdir(parents=True)
    started = time.time()
    meta = dict(
        config=cfg,
        source_hashes=source_manifest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        python=platform.python_version(),
        numpy=np.__version__,
        start_unix=started,
    )
    (out / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows = []
    raw = []
    worlds = []
    for seed in cfg["seeds"]:
        world = World(seed)
        old = world.sample(8)
        later = world.sample(cfg["later_trials"])
        test = world.sample(cfg["test_trials"], heldout=True)
        control_world = World(seed + 10000)
        scrambled = control_world.sample(cfg["later_trials"], scrambled=True)
        scrambled_test = control_world.sample(cfg["test_trials"], heldout=True, scrambled=True)
        train = old * cfg["novice_repeats"] + later * cfg["epochs"]
        world_record = dict(
            seed=seed,
            old=[asdict(t) for t in old],
            later=[asdict(t) for t in later],
            test=[asdict(t) for t in test],
            scrambled=[asdict(t) for t in scrambled],
            scrambled_test=[asdict(t) for t in scrambled_test],
            oracle_mapping=world.mapping.tolist(),
            cues=world.cues.tolist(),
            signals=world.signals.tolist(),
            fillers=world.fillers.tolist(),
        )
        worlds.append(world_record)
        oracle = learned_oracle(later, test)
        best_control = 0
        for rate in cfg["control_learning_rates"]:
            for order in [1, 3]:
                control = NgramDelta(train, rate, max_length=order)
                for t in train:
                    control.train(t.sequence, t.outcome)
                acc = accuracy(control, test)
                if order == 3:
                    best_control = max(best_control, acc)
                rows.append(
                    dict(
                        seed=seed,
                        mode="ngram" if order == 3 else "bag",
                        setting=rate,
                        novel=acc,
                        old=accuracy(control, old),
                        parameters=int(control.w.size),
                        presentations=len(train),
                    )
                )
        # First/last-token majority lookup, learned on identical presentations.
        position = []
        for pos in [0, -1]:
            counts = np.zeros((14, 4))
            for t in train:
                counts[t.sequence[pos], t.outcome] += 1
            position.append(
                float(np.mean([np.argmax(counts[t.sequence[pos]]) == t.outcome for t in test]))
            )
        for vigilance in cfg["vigilances"]:
            for mode in cfg["modes"]:
                model = Learner(seed, vigilance, cfg["feedback"], mode)
                tag_counts = np.zeros((model.capacity, 8))
                for t in old * cfg["novice_repeats"]:
                    model.train(t.sequence, t.outcome)
                    _, r = model.response(t.sequence)
                    if r.any():
                        tag_counts[np.argmax(r), t.tag] += 1
                novice_acc = accuracy(model, old)
                novice_reps = np.array([model.response(t.sequence)[1] for t in old])
                novice_weights = model.w.copy()
                # These are readiness diagnostics, not a Stage2 interpretation.
                for t in later * cfg["epochs"]:
                    model.train(t.sequence, t.outcome)
                    _, r = model.response(t.sequence)
                    if r.any():
                        tag_counts[np.argmax(r), t.tag] += 1
                novel = accuracy(model, test)
                oldacc = accuracy(model, old)
                isolated = float(np.mean([model.predict(x) == y for x, y in world.isolated()]))
                motif_counts = np.zeros((model.capacity, 8))
                contexts = [set() for _ in range(model.capacity)]
                for t in later:
                    _, r = model.response(t.sequence)
                    if r.any():
                        j = int(np.argmax(r))
                        motif_counts[j, t.motif] += 1
                        contexts[j].add(t.context)
                active = motif_counts.sum(1) > 0
                purity = float(motif_counts.max(1).sum() / max(motif_counts.sum(), 1))
                reusable = float(
                    sum(
                        motif_counts[j].sum()
                        for j in range(model.n)
                        if len(contexts[j]) >= 2
                        and motif_counts[j].max() / max(motif_counts[j].sum(), 1) >= 0.8
                    )
                    / max(motif_counts.sum(), 1)
                )
                state_id = f"s{seed}_{vigilance}_{mode}"
                np.savez_compressed(
                    out / (state_id + ".npz"),
                    w=model.w,
                    top=model.top,
                    counts=model.counts,
                    length=model.length,
                    novice_weights=novice_weights,
                    novice_representations=novice_reps,
                    tag_counts=tag_counts,
                )

                def records(trials, split):
                    tags = []
                    for i, t in enumerate(trials):
                        prob, r = model.response(t.sequence)
                        j = int(np.argmax(r)) if r.any() else -1
                        tagpred = int(np.argmax(tag_counts[j])) if j >= 0 else 0
                        tags.append(tagpred == t.tag)
                        raw.append(
                            dict(
                                state=state_id,
                                seed=seed,
                                setting=vigilance,
                                mode=mode,
                                split=split,
                                index=i,
                                outcome=t.outcome,
                                prediction=int(np.argmax(prob)),
                                probability=prob.tolist(),
                                winner=j,
                                motif=t.motif,
                                tag=t.tag,
                                tag_prediction=tagpred,
                            )
                        )
                    return float(np.mean(tags))

                oldtags = records(old, "old")
                records(test, "novel")
                lesioned = float(
                    np.mean(
                        [
                            np.argmax(model.response(t.sequence, lesion=True)[0]) == t.outcome
                            for t in test
                        ]
                    )
                )
                rows.append(
                    dict(
                        seed=seed,
                        mode=mode,
                        setting=vigilance,
                        novel=novel,
                        old=oldacc,
                        novice_old=novice_acc,
                        isolated=isolated,
                        old_tags=oldtags,
                        purity=purity,
                        reuse_fraction=reusable,
                        categories=model.n,
                        active_categories=int(active.sum()),
                        filter_path_length=model.movement,
                        resets=model.resets,
                        overflow=model.overflows,
                        lesion_accuracy=lesioned,
                        parameters=int(model.w.size + model.top.size + model.counts.size),
                        presentations=len(train),
                    )
                )
                print(
                    state_id,
                    "novel",
                    round(novel, 3),
                    "old",
                    round(oldacc, 3),
                    "overflow",
                    model.overflows,
                    flush=True,
                )
        sm = NgramDelta(scrambled * cfg["epochs"], 0.2)
        for t in scrambled * cfg["epochs"]:
            sm.train(t.sequence, t.outcome)
        rows.append(
            dict(
                seed=seed,
                mode="stage0",
                setting=0,
                novel=oracle,
                strong_control=best_control,
                positional=max(position),
                scrambled=accuracy(sm, scrambled_test),
                exact_overlap=len(set(t.sequence for t in later) & set(t.sequence for t in test)),
            )
        )
    (out / "worlds.json").write_text(json.dumps(worlds, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    # Selection uses development worlds only; report entire grid regardless.
    settings = {
        v: np.mean([r["novel"] for r in rows if r["mode"] == "full" and r["setting"] == v])
        for v in cfg["vigilances"]
    }
    chosen = max(settings, key=lambda value: float(settings[value]))
    selected = {
        m: [r for r in rows if r["mode"] == m and r["setting"] == chosen] for m in cfg["modes"]
    }
    full = selected["full"]

    def mean(rs, key):
        return float(np.mean([r[key] for r in rs]))

    stage0 = [r for r in rows if r["mode"] == "stage0"]
    checks = {
        "stage0_oracle": all(r["novel"] == 1 and r["exact_overlap"] == 0 for r in stage0),
        "stage0_control_acquired": all(r["strong_control"] >= 0.9 for r in stage0),
        "stage0_position_nontrivial": all(r["positional"] <= 0.6 for r in stage0),
        "stage0_scrambled": all(r["scrambled"] <= 0.4 for r in stage0),
        "new_and_old_acquired": all(
            r["novel"] >= 0.8 and r["old"] >= 0.85 and r["novice_old"] >= 0.85 for r in full
        ),
        "old_subchunk_retention": all(r["isolated"] >= 0.85 for r in full),
        "filters_learn": all(r["filter_path_length"] > 1 for r in full),
        "learning_advantage": mean(full, "novel") - mean(selected["frozen"], "novel") >= 0.1,
        "selective_reuse": all(r["purity"] >= 0.8 and r["reuse_fraction"] >= 0.8 for r in full),
        "feedback_contribution": mean(full, "novel") - mean(selected["no_feedback"], "novel")
        >= 0.03,
        "predictive_matching_contribution": mean(full, "novel")
        - mean(selected["no_predictive"], "novel")
        >= 0.03,
        "competition_contribution": mean(full, "novel") - mean(selected["no_competition"], "novel")
        >= 0.03,
        "no_capacity_saturation": all(r["overflow"] == 0 for r in full),
    }
    # Novel relational combination / hierarchical composition are not proven by
    # this task, even if every numerical readiness check succeeds.
    ready = all(checks.values())
    decision = dict(
        selected_vigilance=chosen,
        checks=checks,
        stage0=all(v for k, v in checks.items() if k.startswith("stage0")),
        stage1_readiness=ready,
        stage2="not run",
        stage3="not run",
        confirmation="not run",
        reason="Need independent frozen confirmation and hierarchy/relational construct validation"
        if ready
        else "Prerequisite failure: do not interpret recoding or drift",
        reserved_confirmation_seeds=cfg["reserved_confirmation_seeds"],
        comparisons={
            m: paired_interval([r["novel"] for r in full], [r["novel"] for r in selected[m]])
            for m in cfg["modes"]
            if m != "full"
        },
        elapsed_seconds=time.time() - started,
    )
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
