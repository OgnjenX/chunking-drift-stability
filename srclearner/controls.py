"""Trainable finite-context control and disclosed oracle capacity check."""

import numpy as np


class NgramDelta:
    """No chunk categories/feedback: softmax linear prediction of observed ngrams.

    Dictionary is built only from training observations. Same raw token sequences
    and outcomes as the proposed learner; outcome loss adjusts linear weights.
    Includes length 1..3 windows and bias. One online update per presentation.
    """

    def __init__(self, training, learning_rate=0.2, max_length=3):
        self.max_length, self.lr = max_length, learning_rate
        keys = sorted({k for t in training for k in self.keys(t.sequence)})
        self.index = {k: i + 1 for i, k in enumerate(keys)}
        self.w = np.zeros((len(keys) + 1, 4))

    def keys(self, sequence):
        return {
            tuple(sequence[e - length : e])
            for e in range(1, len(sequence) + 1)
            for length in range(1, min(e, self.max_length) + 1)
        }

    def features(self, sequence):
        return [0] + [self.index[k] for k in self.keys(sequence) if k in self.index]

    def train(self, sequence, outcome):
        ids = self.features(sequence)
        z = self.w[ids].sum(0)
        p = np.exp(z - z.max())
        p /= p.sum()
        target = np.eye(4)[outcome]
        self.w[ids] += self.lr * (target - p)

    def predict(self, sequence):
        return int(np.argmax(self.w[self.features(sequence)].sum(0)))


def learned_oracle(training, test):
    """Uses true motif IDs ONLY for disclosed task learnability evaluation."""
    count = np.zeros((8, 4))
    for t in training:
        count[t.motif, t.outcome] += 1
    return float(np.mean([np.argmax(count[t.motif]) == t.outcome for t in test]))
