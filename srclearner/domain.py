"""Experimenter metadata is never sent to the hidden learner."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Trial:
    sequence: tuple
    outcome: int
    motif: int  # evaluation/oracle only
    tag: int  # arbitrary, irrecoverable from the task rule
    context: tuple


class World:
    """Cue-conditioned signal mapping, embedded at variable temporal positions.

    Cue/signal identities are randomly assigned raw vocabulary channels per world.
    Two cues, four signals, eight filler tokens; outcomes are balanced random
    permutations conditional on cue. An extra unmarked signal is a distractor.
    Train/test filler ordered pairs have even/odd parity respectively, but every
    individual filler token occurs in training. No unseen primitive token test.
    """

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        v = self.rng.permutation(14)
        self.cues, self.signals, self.fillers = v[:2], v[2:6], v[6:]
        self.mapping = np.array([self.rng.permutation(4) for _ in range(2)])

    def sample(self, n, heldout=False, scrambled=False):
        trials = []
        for i in range(n):
            cue, signal = i % 2, (i // 2) % 4
            pairs = [
                (a, b) for a in range(8) for b in range(8) if a != b and (a + b) % 2 == int(heldout)
            ]
            a, b = pairs[self.rng.integers(len(pairs))]
            seq = [
                int(self.fillers[a]),
                int(self.fillers[b]),
                int(self.signals[self.rng.integers(4)]),
            ]
            at = self.rng.integers(len(seq) + 1)
            seq[at:at] = [int(self.cues[cue]), int(self.signals[signal])]
            if self.rng.random() < 0.5:
                seq.append(int(self.fillers[self.rng.integers(8)]))
            y = int(self.rng.integers(4)) if scrambled else int(self.mapping[cue, signal])
            trials.append(Trial(tuple(seq), y, cue * 4 + signal, int(self.rng.integers(8)), (a, b)))
        self.rng.shuffle(trials)
        return trials

    def isolated(self):
        return [
            (tuple([int(c), int(s)]), int(self.mapping[a, b]))
            for a, c in enumerate(self.cues)
            for b, s in enumerate(self.signals)
        ]
