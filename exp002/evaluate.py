"""Shared Stage 1 evaluation; fixed heads, trial pairing and explicit prerequisites."""

from copy import deepcopy

import numpy as np

from .controls import GRU, Readout, StoreControl, scores
from .model import TemporalField
from .task import stream, trials


def fit(seed, params, old, long, repetitions=30, gru_lr=0.01):
    """Train all controls on identical old-then-long development streams."""
    field = TemporalField(seed, params)
    initial = field.W.copy()
    head = Readout(len(field.state()))
    nofield = TemporalField(seed, params)
    nohead = Readout(len(nofield.state()))
    store = StoreControl(dt=params.dt)
    gru = GRU(seed, hidden=21, lr=gru_lr)
    phases = []
    # Identical sensory streams, order, labels, and trial counts for each model.
    # Phase B has ONLY longer sequences; no hidden replay of old sensory inputs.
    for phase, seqs in [(1, old), (2, long)]:
        losses = []
        for seq, a in trials(
            seqs, params, seed, repetitions, {"delay": 2.0, "noise": 0.005}, phase
        ):
            h = field.run(a, learn=True)[-1]
            head.update(h, seq)
            h = nofield.run(a, learn=False, chunking=False)[-1]
            nohead.update(h, seq)
            store.head.update(store.run(a), seq)
            losses.append(gru.update(a, seq))
        models = {
            "field": field,
            "head": head,
            "no_field": nofield,
            "no_head": nohead,
            "store": store,
            "gru": gru,
        }
        clean = evaluate(models, old + long, seed, {"delay": 0.5}, repeats=1, phase=70 + phase)
        phases.append(
            {
                "phase": phase,
                "clean": clean,
                "gru_loss_mean": float(np.mean(losses)),
                "old_probes": clean_states(field, old),
                "W": field.W.copy(),
                "M": field.M.copy(),
            }
        )
    return models, phases, initial


def clean_states(field, seqs):
    """Probe ordered sequences under the fixed clean diagnostic condition."""
    return np.array(
        [field.run(stream(s, field.p, np.random.default_rng(0), delay=0.5))[-1] for s in seqs]
    )


def evaluate(models, seqs, seed, difficulty, repeats=8, phase=99, ablations=False):
    """Measure each model and optional interventions with paired sensory trials."""
    results = []
    p = models["field"].p
    variants = ["intact"] + (
        ["feedback_off", "competition_off", "chunks_off", "chunk_lesion"] if ablations else []
    )
    lesion = deepcopy(models["field"])
    rng = np.random.default_rng([seed, 719])
    idx = rng.choice(len(lesion.mask), len(lesion.mask) // 4, False)
    lesion.mask[idx] = 0
    for rep in range(repeats):
        for i, seq in enumerate(seqs):
            a = stream(seq, p, np.random.default_rng([seed, phase, rep, i]), **difficulty)
            h = models["field"].run(a)[-1]
            row = {
                "seed": seed,
                "sequence": seq,
                "rep": rep,
                "model": "field",
                "intervention": "intact",
                **scores(models["head"].predict(h), seq),
                "state": h,
            }
            results.append(row)
            for variant in variants[1:]:
                opts = {
                    "feedback": variant != "feedback_off",
                    "competition": variant != "competition_off",
                    "chunking": variant != "chunks_off",
                }
                f = lesion if variant == "chunk_lesion" else models["field"]
                hh = f.run(a, **opts)[-1]
                results.append(
                    {
                        "seed": seed,
                        "sequence": seq,
                        "rep": rep,
                        "model": "field",
                        "intervention": variant,
                        **scores(models["head"].predict(hh), seq),
                        "state": hh,
                    }
                )
            h = models["no_field"].run(a, chunking=False)[-1]
            results.append(
                {
                    "seed": seed,
                    "sequence": seq,
                    "rep": rep,
                    "model": "nonchunk_wm",
                    "intervention": "intact",
                    **scores(models["no_head"].predict(h), seq),
                    "state": h,
                }
            )
            results.append(
                {
                    "seed": seed,
                    "sequence": seq,
                    "rep": rep,
                    "model": "store2",
                    "intervention": "intact",
                    **scores(models["store"].head.predict(models["store"].run(a)), seq),
                }
            )
            out, h, _ = models["gru"].forward(a)
            results.append(
                {
                    "seed": seed,
                    "sequence": seq,
                    "rep": rep,
                    "model": "gru",
                    "intervention": "intact",
                    **scores(out, seq),
                    "state": h,
                }
            )
    return results


def summary(rows):
    """Aggregate intact performance separately for each model."""
    return {
        m: {
            k: float(
                np.mean([r[k] for r in rows if r["model"] == m and r["intervention"] == "intact"])
            )
            for k in ("exact", "token", "length")
        }
        for m in sorted(set(r["model"] for r in rows))
    }


def serialize(obj):
    """Convert NumPy measurements for JSON, rejecting unsupported objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))
