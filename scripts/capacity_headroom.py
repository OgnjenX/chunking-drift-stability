"""Second, explicitly exploratory attempt; see protocols/capacity-headroom-v2.md."""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from srclearner.controls import NgramDelta, learned_oracle
from srclearner.domain import Trial, World
from srclearner.mechanisms import Learner
from srclearner.metrics import accuracy


def main():
    out = Path("results/development-v2")
    if out.exists():
        raise SystemExit("Refusing overwrite")
    out.mkdir(parents=True)
    started = time.time()
    sources = list(Path("srclearner").glob("*.py")) + [
        Path(__file__),
        Path("protocols/capacity-headroom-v2.md"),
    ]
    manifest = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    provenance = dict(
        source_hashes=manifest,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        purpose="Exploratory capacity/headroom follow-up, not confirmation",
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    modes = [
        "full",
        "frozen",
        "no_chunk",
        "no_feedback",
        "no_predictive",
        "no_competition",
        "ngram",
    ]
    rows, raw, worlds = [], [], []
    for seed in [311, 312, 313]:
        w = World(seed)
        old = w.sample(8)
        later = w.sample(320)
        test = w.sample(160, heldout=True)
        train = old * 8 + later * 4
        rng = np.random.default_rng(seed + 50000)
        conditions = {0.0: test}
        for noise in [0.1, 0.25]:
            conditions[noise] = [
                Trial(
                    tuple(
                        int(rng.integers(14)) if rng.random() < noise else token
                        for token in t.sequence
                    ),
                    t.outcome,
                    t.motif,
                    t.tag,
                    t.context,
                )
                for t in test
            ]
        worlds.append(
            dict(
                seed=seed,
                old=[asdict(t) for t in old],
                later=[asdict(t) for t in later],
                conditions={str(k): [asdict(t) for t in v] for k, v in conditions.items()},
                oracle=learned_oracle(later, test),
            )
        )
        for mode in modes:
            start = time.time()
            if mode == "ngram":
                model = NgramDelta(train, 0.2)
            else:
                model = Learner(seed, 0.94, 0.25, mode, capacity=1536)
            for t in old * 8:
                model.train(t.sequence, t.outcome)
            novice = accuracy(model, old)
            novice_tags = None
            if isinstance(model, Learner):
                tags = np.zeros((1536, 8))
                for t in old * 8:
                    _, r = model.response(t.sequence)
                    if r.any():
                        tags[np.argmax(r), t.tag] += 1
                novice_tags = float(
                    np.mean(
                        [
                            np.argmax(tags[np.argmax(model.response(t.sequence)[1])]) == t.tag
                            for t in old
                        ]
                    )
                )
            for t in later * 4:
                model.train(t.sequence, t.outcome)
                if isinstance(model, Learner):
                    _, r = model.response(t.sequence)
                    if r.any():
                        tags[np.argmax(r), t.tag] += 1
            old_tags = None
            if isinstance(model, Learner):
                old_tags = float(
                    np.mean(
                        [
                            np.argmax(tags[np.argmax(model.response(t.sequence)[1])]) == t.tag
                            for t in old
                        ]
                    )
                )
                np.savez_compressed(
                    out / f"s{seed}_{mode}.npz",
                    w=model.w,
                    top=model.top,
                    counts=model.counts,
                    length=model.length,
                    tags=tags,
                )
            else:
                np.savez_compressed(out / f"s{seed}_{mode}.npz", w=model.w)
                (out / f"s{seed}_{mode}_dictionary.json").write_text(
                    json.dumps([[list(k), v] for k, v in model.index.items()])
                )
            for noise, trials in conditions.items():
                predictions = [model.predict(t.sequence) for t in trials]
                for i, (t, pred) in enumerate(zip(trials, predictions)):
                    raw.append(
                        dict(
                            seed=seed,
                            mode=mode,
                            noise=noise,
                            index=i,
                            outcome=t.outcome,
                            prediction=pred,
                        )
                    )
                row = dict(
                    seed=seed,
                    mode=mode,
                    noise=noise,
                    accuracy=float(
                        np.mean([pred == t.outcome for pred, t in zip(predictions, trials)])
                    ),
                    old=accuracy(model, old),
                    novice_old=novice,
                    novice_tags=novice_tags,
                    old_tags=old_tags,
                    overflow=model.overflows if isinstance(model, Learner) else 0,
                    categories=model.n if isinstance(model, Learner) else None,
                    learned_parameters=int(model.w.size * 2 + model.counts.size)
                    if isinstance(model, Learner)
                    else int(model.w.size),
                    elapsed_seconds=time.time() - start,
                )
                rows.append(row)
                print(
                    seed,
                    mode,
                    noise,
                    round(row["accuracy"], 3),
                    "overflow",
                    row["overflow"],
                    flush=True,
                )
    (out / "worlds.json").write_text(json.dumps(worlds, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    full = [r for r in rows if r["mode"] == "full" and r["noise"] == 0]
    degraded = [r for r in rows if r["mode"] == "full" and r["noise"] == 0.1]
    feedback = [r for r in rows if r["mode"] == "no_feedback" and r["noise"] == 0.1]
    checks = dict(
        capacity_resolved=all(r["overflow"] == 0 for r in full),
        novice_acquired=all(r["novice_old"] >= 0.85 for r in full),
        new_and_old=all(r["accuracy"] >= 0.8 and r["old"] >= 0.85 for r in full),
        degraded_headroom=all(0.35 <= r["accuracy"] <= 0.9 for r in degraded),
        degraded_feedback_effect=np.mean([r["accuracy"] for r in degraded])
        - np.mean([r["accuracy"] for r in feedback])
        >= 0.03,
    )
    decision = dict(
        checks={k: bool(v) for k, v in checks.items()},
        status="exploratory no-go"
        if not all(checks.values())
        else "needs independent full Stage1 protocol",
        confirmation="not run",
        stage2="not run",
        stage3="not run",
        elapsed_seconds=time.time() - started,
    )
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
