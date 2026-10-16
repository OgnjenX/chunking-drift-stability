"""Fixed grammar and explicitly separated training/evaluation randomness."""

import numpy as np

ALPHABET = "ABCDEFGH"
PARTS = ["AB", "CD", "EF", "GH"]
ITEMS = list(ALPHABET)
PAIRS = PARTS.copy()
TRAIN_COMBINATIONS = [(i, j) for i in range(4) for j in (i, (i + 1) % 4)]
NEW = [PARTS[i] + PARTS[j] for i, j in TRAIN_COMBINATIONS]
HELDOUT = [
    PARTS[i] + PARTS[j] for i in range(4) for j in range(4) if (i, j) not in TRAIN_COMBINATIONS
]
OLD = ITEMS + PAIRS
LEARNED = OLD + NEW


def encode(sequences):
    """Encode ordered sequences with explicit padding positions."""
    target = np.full((len(sequences), 4), 8, dtype=int)
    for i, seq in enumerate(sequences):
        for pos, symbol in enumerate(seq):
            target[i, pos] = ALPHABET.index(symbol)
    return np.eye(9)[target].reshape(len(sequences), 36), target


def corrupt(sequences, rng, missing=1, repeats=1, probability=1.0):
    """Mask sampled item positions with reproducible, separate probe randomness."""
    seqs = [s for s in sequences for _ in range(repeats)]
    x, target = encode(seqs)
    view = x.reshape(-1, 4, 9)
    for i, seq in enumerate(seqs):
        if rng.random() < probability:
            positions = rng.choice(len(seq), min(missing, len(seq)), replace=False)
            view[i, positions] = 0.0
    return x, target
