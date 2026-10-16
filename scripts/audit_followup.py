"""Read saved predictions independently of the training runner; replay one learning path."""

import hashlib
import json
from pathlib import Path

import numpy as np

from srclearner.controls import NgramDelta
from srclearner.domain import World
from srclearner.mechanisms import Learner, cosine_matrix, working_memory


def main():
    p = Path("results/development-v2")
    meta = json.loads((p / "provenance.json").read_text())
    for name, digest in meta["source_hashes"].items():
        f = Path(name)
        if not f.exists() and f.name == "capacity_headroom.py":
            f = Path("scripts") / f.name
        assert hashlib.sha256(f.read_bytes()).hexdigest() == digest
    worlds = {w["seed"]: w for w in json.loads((p / "worlds.json").read_text())}
    episodic = []
    for seed, world in worlds.items():
        lookup = {tuple(t["sequence"]): t["tag"] for t in world["old"]}
        score = float(np.mean([lookup[tuple(t["sequence"])] == t["tag"] for t in world["old"]]))
        assert score == 1.0
        episodic.append(
            {
                "seed": seed,
                "exact_lookup_old_tag_sanity": score,
                "scope": "Disclosed capacity control, never used by the learner",
            }
        )
    (p / "episodic_sanity.json").write_text(json.dumps(episodic, indent=2) + "\n")
    rows = json.loads((p / "metrics.json").read_text())
    models = {}
    buckets = {}
    checked = 0
    for line in (p / "trials.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["seed"], r["mode"])
        if key not in models:
            state = np.load(p / f"s{key[0]}_{key[1]}.npz")
            if key[1] == "ngram":
                model = NgramDelta([], 0.2)
                dictionary = json.loads((p / f"s{key[0]}_ngram_dictionary.json").read_text())
                model.index = {tuple(k): v for k, v in dictionary}
                model.w = state["w"]
            else:
                model = Learner(key[0], 0.94, 0.25, key[1], capacity=1536)
                for name in ["w", "top", "counts", "length"]:
                    setattr(model, name, state[name])
                model.n = int(np.count_nonzero(model.length))
            models[key] = model
        t = worlds[key[0]]["conditions"][str(r["noise"])][r["index"]]
        assert models[key].predict(t["sequence"]) == r["prediction"]
        assert t["outcome"] == r["outcome"]
        buckets.setdefault((*key, r["noise"]), []).append(r["prediction"] == r["outcome"])
        checked += 1
    for r in rows:
        assert np.mean(buckets[r["seed"], r["mode"], r["noise"]]) == r["accuracy"]
    # Replay the complete learning path for one original world using raw streams.
    v1 = Path("results/development-v1")
    world = json.loads((v1 / "worlds.json").read_text())[0]
    replay = Learner(world["seed"], 0.94, 0.25, "full")
    for t in world["old"] * 8 + world["later"] * 4:
        replay.train(t["sequence"], t["outcome"])
    saved = np.load(v1 / "s211_0.94_full.npz")
    for name in ["w", "top", "counts", "length"]:
        np.testing.assert_array_equal(getattr(replay, name), saved[name])
    # Evaluate selected-category structure without sending motif truth to learner.
    structure = []
    for seed in worlds:
        world = World(seed)
        model = models[seed, "full"]
        counts = np.zeros((1536, 8))
        context = [set() for _ in range(1536)]
        tuning = []
        for t in worlds[seed]["conditions"]["0.0"]:
            _, r = model.response(t["sequence"])
            j = int(np.argmax(r))
            counts[j, t["motif"]] += 1
            context[j].add(tuple(t["context"]))
            c, s = divmod(t["motif"], 4)
            target = working_memory([int(world.cues[c]), int(world.signals[s])])
            tuning.append(float(cosine_matrix(model.w[j : j + 1], target)[0]))
        structure.append(
            dict(
                seed=seed,
                active_categories=int(np.count_nonzero(counts.sum(1))),
                motif_purity=float(counts.max(1).sum() / counts.sum()),
                multi_context_fraction=float(
                    sum(counts[j].sum() for j in range(1536) if len(context[j]) >= 2) / counts.sum()
                ),
                selected_filter_true_motif_cosine=float(np.mean(tuning)),
            )
        )
    (p / "structure.json").write_text(json.dumps(structure, indent=2) + "\n")
    summary = dict(
        reconstructed_predictions=checked,
        models=len(models),
        aggregates_match=True,
        source_hashes_match=True,
        original_learning_path_replayed_bitwise=True,
        scope="Implementation/data reconstruction, not independent biological validation",
    )
    (p / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
