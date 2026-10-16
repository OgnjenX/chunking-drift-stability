"""Source equation plus explicit discrete approximations (see docs/mechanism.md)."""

import numpy as np


def instar(w, x, rate=0.2, gate=1.0, duration=1.0):
    """Exact constant-input solution of KG2014 Eq13, not forward Euler."""
    total = np.sum(x)
    if total <= 0:
        return w.copy()
    return x / total + (w - x / total) * np.exp(-rate * gate * total * duration)


def working_memory(sequence, vocab=14, decay=0.65):
    """Primacy insertion; existing nonzero ratios survive append; repeats accumulate.

    Discrete design choice, NOT STORE2 differential equations. Stimulus onset
    and end of each candidate window are externally supplied by the scanner.
    """
    state = np.zeros(vocab)
    for rank, token in enumerate(sequence):
        state[token] += decay**rank
    return state / max(state.sum(), 1e-12)


def candidates(sequence, max_length=3):
    for end in range(1, len(sequence) + 1):
        for length in range(1, min(end, max_length) + 1):
            yield length, working_memory(sequence[end - length : end])


def cosine_matrix(w, x):
    return (w @ x) / np.maximum(np.linalg.norm(w, axis=1) * np.linalg.norm(x), 1e-12)


def fuzzy_art_choice_match_learn(x, w, alpha=0.01, beta=1.0):
    """Independent reference of fuzzy ART operations, not the task learner.

    Caller supplies complement-coded x; choice Eq8, match Eq9, update Eq11
    in Carpenter/Grossberg fuzzy ART algorithm (BU source in source manifest).
    """
    intersection = np.minimum(x, w)
    return (
        intersection.sum() / (alpha + w.sum()),
        intersection.sum() / x.sum(),
        beta * intersection + (1 - beta) * w,
    )


class Learner:
    """Novelty recruitment + local competitive instar + outcome outstar/counts.

    No rule IDs, sequence IDs, relevant positions or truth templates are passed.
    Outcomes enter only conditional category counts; their entropy controls
    predictive selection and (if enabled) mismatch search during training.
    """

    def __init__(self, seed=0, vigilance=0.96, feedback=0.25, mode="full", capacity=384):
        self.rng = np.random.default_rng(seed)
        self.vigilance, self.feedback, self.mode = vigilance, feedback, mode
        self.capacity = capacity
        self.w = np.zeros((capacity, 14))
        self.top = self.w.copy()
        self.length = np.zeros(capacity, int)
        self.counts = np.zeros((capacity, 4))
        self.n = 0
        self.movement = 0.0
        self.resets = 0
        self.overflows = 0
        if mode == "frozen":
            # Generic random filters: no stimulus-dependent recruitment or tuning.
            self.n = capacity
            self.length[:] = np.arange(capacity) % 3 + 1
            self.w[:] = self.rng.dirichlet(np.ones(14) * 0.3, capacity)
            self.top[:] = self.w
        self.initial = self.w.copy()

    def _matches(self, length, x):
        ids = np.flatnonzero(self.length[: self.n] == length)
        if not len(ids):
            return ids, np.empty(0)
        sim = cosine_matrix(self.w[ids], x)
        if self.mode != "no_feedback":
            # Learned template reconnection; no oracle imputation. Two-pass match.
            best = ids[np.argmax(sim)]
            modulated = (1 - self.feedback) * x + self.feedback * self.top[best]
            sim = cosine_matrix(self.w[ids], modulated)
        return ids, sim

    def train(self, sequence, outcome):
        for length, x in candidates(sequence, 1 if self.mode == "no_chunk" else 3):
            ids, sim = self._matches(length, x)
            winner = None
            recruited = False
            if len(ids):
                for k in np.argsort(-sim, kind="stable"):
                    j = ids[k]
                    if sim[k] < self.vigilance and self.mode != "frozen":
                        break
                    n = self.counts[j].sum()
                    # Observed sensory outcome can reject a confident prediction.
                    if (
                        self.mode not in ("no_predictive", "frozen")
                        and n >= 5
                        and self.counts[j].max() / n > 0.8
                        and np.argmax(self.counts[j]) != outcome
                    ):
                        self.resets += 1
                        continue
                    winner = j
                    break
            if winner is None:
                if self.n == self.capacity:
                    self.overflows += 1
                    continue
                recruited = True
                winner = self.n
                self.n += 1
                self.length[winner] = length
                self.w[winner] = np.ones(14) / 14
                self.top[winner] = self.w[winner]
            if self.mode != "frozen":
                old = self.w[winner].copy()
                self.w[winner] = instar(old, x, rate=0.8, duration=4 if recruited else 1)
                self.top[winner] = instar(self.top[winner], x, rate=0.4)
                self.movement += float(np.linalg.norm(old - self.w[winner]))
            self.counts[winner, outcome] += 1

    def response(self, sequence, lesion=False):
        evidence = np.zeros(self.capacity)
        if lesion:
            return np.ones(4) / 4, evidence
        for length, x in candidates(sequence, 1 if self.mode == "no_chunk" else 3):
            ids, sim = self._matches(length, x)
            if not len(ids):
                continue
            n = self.counts[ids].sum(1)
            p = (self.counts[ids] + 1) / (n[:, None] + 4)
            confidence = p.max(1) - 0.25
            # Smoothed reliability/support avoids promoting single-exposure noise.
            utility = confidence * np.log1p(n) if self.mode != "no_predictive" else np.log1p(n)
            score = np.maximum(sim, 0) ** 8 * utility
            if self.mode != "no_competition":
                k = int(np.argmax(score))
                evidence[ids[k]] = max(evidence[ids[k]], float(score[k]))
            else:
                evidence[ids] += score
        if evidence.max() <= 0:
            return np.ones(4) / 4, evidence
        if self.mode != "no_competition":
            j = int(np.argmax(evidence))
            out = (self.counts[j] + 1) / (self.counts[j].sum() + 4)
            response = np.zeros(self.capacity)
            response[j] = evidence[j]
            return out, response
        p = (self.counts + 1) / (self.counts.sum(1)[:, None] + 4)
        return evidence @ p / evidence.sum(), evidence

    def predict(self, sequence):
        return int(np.argmax(self.response(sequence)[0]))
