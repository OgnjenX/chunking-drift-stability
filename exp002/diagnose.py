"""Post-development mechanism diagnostics, not confirmatory hypothesis tests."""

import json
from itertools import permutations
from pathlib import Path

import numpy as np

from .controls import GRU, Readout, scores
from .develop_probe import DEV_LONG, DEV_OLD
from .evaluate import serialize
from .model import Params, TemporalField
from .task import LONG, stream


def main():
    """Measure emergence, numerical refinement, and held-out development cues."""
    root = Path("exp002/results/development-calibration")
    out = root / "diagnostics"
    out.mkdir(exist_ok=False)
    decision = json.loads((root / "decision.json").read_text())
    p = Params(**decision["parameters"])
    rows = []
    mechanisms = []
    for seed in [101, 102, 103]:
        saved = np.load(root / f"s{seed}_field.npz")
        heads = np.load(root / f"s{seed}_readouts.npz")
        m = TemporalField(seed, p)
        m.W = saved["W"].copy()
        m.M = saved["M"].copy()
        initial = TemporalField(seed, p)
        initial.W = heads["initial_W"].copy()
        initial.M = initial.W.copy()
        head = Readout(len(m.state()))
        head.B = heads["field"].copy()
        wm = TemporalField(seed, p)
        wmhead = Readout(len(wm.state()))
        wmhead.B = heads["wm"].copy()
        gru = GRU(seed, hidden=21)
        gru.w = {k: heads[k].copy() for k in gru.w}
        all_states = []
        initial_states = []
        winner_labels = []
        for i, seq in enumerate(DEV_OLD + DEV_LONG):
            a = stream(seq, p, np.random.default_rng([seed, 922, i]), delay=2.0, noise=0.005)
            trace = m.run(a)
            s = trace[-1]
            all_states.append(s)
            before = initial.run(a)[-1]
            initial_states.append(before)
            c = s[8:]
            winner_labels.append(
                {
                    "sequence": seq,
                    "winner": int(c.argmax()),
                    "winner_size": float(m.size[c.argmax()]),
                    "active_over_threshold": int(np.sum(c > 0.2)),
                }
            )
            for intervention in [
                "intact",
                "feedback_off",
                "competition_off",
                "chunks_off",
                "chunk_lesion",
            ]:
                old_mask = m.mask.copy()
                if intervention == "chunk_lesion":
                    idx = np.random.default_rng([seed, 719]).choice(
                        len(m.mask), len(m.mask) // 4, False
                    )
                    m.mask[idx] = 0
                h = m.run(
                    a,
                    feedback=intervention != "feedback_off",
                    competition=intervention != "competition_off",
                    chunking=intervention != "chunks_off",
                )[-1]
                m.mask = old_mask
                rows.append(
                    {
                        "seed": seed,
                        "condition": "training_matched",
                        "sequence": seq,
                        "intervention": intervention,
                        "model": "field",
                        **scores(head.predict(h), seq),
                    }
                )
            h = wm.run(a, chunking=False)[-1]
            rows.append(
                {
                    "seed": seed,
                    "condition": "training_matched",
                    "sequence": seq,
                    "intervention": "intact",
                    "model": "nonchunk_wm",
                    **scores(wmhead.predict(h), seq),
                }
            )
            out_logits, _, _ = gru.forward(a)
            rows.append(
                {
                    "seed": seed,
                    "condition": "training_matched",
                    "sequence": seq,
                    "intervention": "intact",
                    "model": "gru",
                    **scores(out_logits, seq),
                }
            )
        same_set = []
        for i, a in enumerate(DEV_LONG):
            for j, b in enumerate(DEV_LONG[:i]):
                if set(a) == set(b):
                    ca = all_states[len(DEV_OLD) + i][8:]
                    cb = all_states[len(DEV_OLD) + j][8:]
                    same_set.append(float(ca.argmax() != cb.argmax()))
        mechanisms.append(
            {
                "seed": seed,
                "winner_count": len(set(x["winner"] for x in winner_labels)),
                "sequence_count": len(winner_labels),
                "distinct_winner_fraction": len(set(x["winner"] for x in winner_labels))
                / len(winner_labels),
                "same_set_order_winner_discrimination": float(np.mean(same_set)),
                "mean_active_chunks": float(
                    np.mean([x["active_over_threshold"] for x in winner_labels])
                ),
                "filter_relative_change": float(
                    np.linalg.norm(m.W - heads["initial_W"]) / np.linalg.norm(heads["initial_W"])
                ),
                "labels": winner_labels,
            }
        )
        # Novel development-only compositions, excluding every confirmation long list.
        novel = [
            "".join(t)
            for k in (3, 4)
            for t in permutations("ABCD", k)
            if "".join(t) not in DEV_LONG + LONG
        ][:16]
        for i, seq in enumerate(novel):
            a = stream(seq, p, np.random.default_rng([seed, 923, i]), delay=2.0, noise=0.005)
            h = m.run(a)[-1]
            rows.append(
                {
                    "seed": seed,
                    "condition": "novel_development_composition",
                    "sequence": seq,
                    "intervention": "intact",
                    "model": "field",
                    **scores(head.predict(h), seq),
                }
            )
            out_logits, _, _ = gru.forward(a)
            rows.append(
                {
                    "seed": seed,
                    "condition": "novel_development_composition",
                    "sequence": seq,
                    "intervention": "intact",
                    "model": "gru",
                    **scores(out_logits, seq),
                }
            )
        np.savez_compressed(
            out / f"s{seed}_codes.npz",
            learned=np.array(all_states),
            initial=np.array(initial_states),
            W=m.W,
            M=m.M,
        )
        a = stream("ACDB", p, np.random.default_rng([seed, 924]), delay=2.0)
        np.savez_compressed(out / f"s{seed}_temporal.npz", stimulus=a, trace=m.run(a), dt=p.dt)
    (out / "trials.json").write_text(json.dumps(rows, indent=2, default=serialize) + "\n")
    (out / "mechanisms.json").write_text(json.dumps(mechanisms, indent=2, default=serialize) + "\n")
    print(json.dumps([{k: v for k, v in a.items() if k != "labels"} for a in mechanisms], indent=2))


if __name__ == "__main__":
    main()
