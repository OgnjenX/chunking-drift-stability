"""Reconstruct saved trial predictions and raw aggregates from archived weights."""

import json
from pathlib import Path

import numpy as np

from .mechanisms import Learner
from .run import source_manifest


def main():
    root = Path("results/development-v1")
    meta = json.loads((root / "provenance.json").read_text())
    assert meta["source_hashes"] == source_manifest(), "Source/config/protocol changed"
    worlds = {w["seed"]: w for w in json.loads((root / "worlds.json").read_text())}
    rows = json.loads((root / "metrics.json").read_text())
    cache = {}
    aggregates = {}
    checked = 0
    for line in (root / "trials.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = record["state"]
        if key not in cache:
            saved = np.load(root / (key + ".npz"))
            m = Learner(
                record["seed"], record["setting"], meta["config"]["feedback"], record["mode"]
            )
            for name in ["w", "top", "counts", "length"]:
                setattr(m, name, saved[name].copy())
            m.n = int(np.count_nonzero(m.length))
            cache[key] = m
        trial = worlds[record["seed"]]["old" if record["split"] == "old" else "test"][
            record["index"]
        ]
        m = cache[key]
        probability, _ = m.response(trial["sequence"])
        np.testing.assert_allclose(probability, record["probability"], rtol=0, atol=1e-14)
        assert int(np.argmax(probability)) == record["prediction"]
        assert trial["outcome"] == record["outcome"]
        bucket = (record["seed"], record["mode"], record["setting"], record["split"])
        aggregates.setdefault(bucket, []).append(record["outcome"] == record["prediction"])
        checked += 1
    for r in rows:
        if r["mode"] not in meta["config"]["modes"]:
            continue
        for split in ["old", "novel"]:
            actual = np.mean(aggregates[r["seed"], r["mode"], r["setting"], split])
            assert actual == r[split]
    for world in worlds.values():
        assert not (
            set(tuple(t["sequence"]) for t in world["later"])
            & set(tuple(t["sequence"]) for t in world["test"])
        )
        assert not (
            set(tuple(t["context"]) for t in world["later"])
            & set(tuple(t["context"]) for t in world["test"])
        )
    reference = json.loads(Path("results/reference-v1/summary.json").read_text())
    assert reference["max_analytic_error"] < 1e-12
    assert reference["end_error"] < 0.003
    result = dict(
        reconstructed_predictions=checked,
        models=len(cache),
        aggregates_match=True,
        split_disjointness=True,
        source_hashes_match=True,
        reference_law=True,
        scope="Readout reconstruction and source-law checks, not independent model validation",
    )
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
