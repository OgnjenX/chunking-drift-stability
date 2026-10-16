"""Readout behavior, coordinate changes, and geometry are separate measures."""

from typing import TypedDict

import numpy as np

from .data import corrupt


class Accuracy(TypedDict):
    """Always-defined clean sequence scores."""

    token: float
    exact: float


class ProbeAccuracy(Accuracy):
    """Probe scores, with no masked score when a cue has no masked items."""

    masked_token: float | None


def accuracy(model, x, target, decoder=None) -> Accuracy:
    """Measure clean token and whole-sequence accuracy, excluding padding."""
    pred = model.forward(x, decoder=decoder)["logits"].argmax(axis=-1)
    valid = target != 8
    return {
        "token": float((pred == target)[valid].mean()),
        "exact": float(np.all((pred == target) | ~valid, axis=1).mean()),
    }


def probe(model, sequences, missing, repeats=32, decoder=None) -> ProbeAccuracy:
    # Same sequence/order/trials across models, schedules, times, and seeds.
    """Score paired corrupted cues, including accuracy only on masked targets."""
    x, target = corrupt(
        sequences, np.random.default_rng(801 + missing), missing=missing, repeats=repeats
    )
    masked = (x.reshape(-1, 4, 9).sum(axis=-1) == 0) & (target != 8)
    pred = model.forward(x, decoder=decoder)["logits"].argmax(axis=-1)
    return {
        **accuracy(model, x, target, decoder),
        "masked_token": float((pred == target)[masked].mean()) if masked.any() else None,
    }


def cosine_rows(a, b):
    """Compute rowwise cosine similarity, treating two silent rows as identical."""
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denom = na * nb
    sim = np.divide((a * b).sum(axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-15)
    sim[(na < 1e-15) & (nb < 1e-15)] = 1.0
    return np.clip(sim, -1, 1)


def change(a, b):
    """Separate coordinate change, relative amplitude change, and linear CKA."""
    if a.shape[1] == 0:
        return {"cosine": None, "relative_rms": None, "one_minus_cka": None}
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    ka, kb = ac @ ac.T, bc @ bc.T
    denom = np.linalg.norm(ka) * np.linalg.norm(kb)
    cka = float((ka * kb).sum() / denom) if denom > 1e-15 else float(np.allclose(a, b))
    return {
        "cosine": float(np.mean(1 - cosine_rows(a, b))),
        "relative_rms": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-15)),
        "one_minus_cka": float(np.clip(1 - cka, 0, 1)),
    }


def structure_contrast(a, b, sequences):
    """Compare recoding similarity for overlapping and disjoint item sets."""
    delta = b - a
    shared, disjoint = [], []
    for i, sequence in enumerate(sequences):
        for j in range(i):
            value = float(cosine_rows(delta[i : i + 1], delta[j : j + 1])[0])
            (shared if set(sequence) & set(sequences[j]) else disjoint).append(value)
    return float(np.mean(shared) - np.mean(disjoint))


def code_stats(c):
    """Report winner coverage and normalized activity entropy."""
    if c.shape[1] == 0:
        return {"winner_count": None, "normalized_entropy": None}
    p = np.abs(c)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-15)
    return {
        "winner_count": int(len(np.unique(c.argmax(axis=1)))),
        "normalized_entropy": float(
            -(p * np.log(p + 1e-300)).sum(axis=1).mean() / np.log(c.shape[1])
        ),
    }
