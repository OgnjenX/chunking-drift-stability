"""Source-traceable temporal circuits. Fidelity decisions: docs/source-audit.md."""

from dataclasses import asdict, dataclass
from itertools import combinations, permutations

import numpy as np


def positive_sigmoid(x, half=1.0):
    """Evaluate the source thresholded quadratic sigmoid."""
    p = np.maximum(x, 0)
    return p * p / (half * half + p * p)


def store2_rhs(x, y, stimulus, gain=0.01):
    """Transcribe the two STORE2 derivatives and complementary sensory gating."""
    total = float(stimulus.sum())
    if total > 1 + 1e-12:
        raise ValueError("STORE2 reference expects at most one unit input")
    return ((gain * stimulus + y - x * x.sum() - 0.7 * x) * total, 5 * (x - y) * (1 - total))


def shunt(value, excitation, inhibition, dt, decay=0.1, lower=0.0, speed=1.0):
    """Exact scalar solution for conductances frozen within one integration step."""
    rate = decay + excitation + inhibition
    equilibrium = (excitation - lower * inhibition) / np.maximum(rate, 1e-15)
    return equilibrium + (value - equilibrium) * np.exp(-speed * dt * rate)


def habituate(z, activity, dt):
    """Integrate a habituative transmitter with activity fixed for one step."""
    rate = 0.01 + 0.1 * activity + 3 * activity * activity
    eq = 0.01 / rate
    return eq + (z - eq) * np.exp(-dt * rate)


@dataclass
class Params:
    """Circuit constants and temporal discretization used by development runs."""

    n_items: int = 4
    max_length: int = 4
    dt: float = 0.05
    pulse: float = 2.0
    gap: float = 0.5
    settle: float = 2.0
    feedback: float = 8.0
    feedback_threshold: float = 0.2
    learning_rate: float = 0.2
    input_gain: float = 0.1
    wm_recurrence: float = 0.7
    x_recurrence: float = 0.05
    field_gain: float = 70.0
    inhibition: float = 1.0


class TemporalField:
    """Source-inspired temporal working memory and competitive list field."""

    def __init__(self, seed=0, params=None):
        self.p = Params() if params is None else params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        supports = []
        weights = []
        for k in range(1, self.p.max_length + 1):
            r = self.rng.uniform(size=k)
            r /= r.sum()
            fluct = 0 if k == 1 else (3 / (10 * np.sqrt(3))) * np.sqrt((k + 1) / (k - 1))
            base = (1 - fluct) / k + fluct * r
            for subset in combinations(range(self.p.n_items), k):
                for values in permutations(base):
                    mask = np.zeros(self.p.n_items)
                    mask[list(subset)] = 1
                    w = np.zeros(self.p.n_items)
                    w[list(subset)] = values
                    supports.append(mask)
                    weights.append(w)
        self.support = np.array(supports)
        self.size = self.support.sum(axis=1)
        self.W = np.array(weights)
        self.M = self.W.copy()
        self.mask = np.ones(len(self.W))
        weighted = self.size[:, None] * self.support
        self.denom = self.size.sum() + self.support @ weighted.sum(axis=0)
        self.reset()

    @property
    def parameter_count(self):
        """Count the bottom-up and top-down learned filter parameters."""
        return int(self.W.size + self.M.size)

    def reset(self):
        """Reset trial activity and transmitter states while retaining learned filters."""
        n = self.p.n_items
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.zi = np.ones(n)
        self.zw = np.ones(n)
        self.c = np.zeros(len(self.W))
        self.zc = np.ones(len(self.W))

    def masking(self, signal):
        """Evaluate the normalized overlap-dependent lateral inhibitory field."""
        source = signal * self.size
        return (source.sum() + self.support @ (self.support.T @ source)) / self.denom

    def step(self, stimulus, learn=False, feedback=True, competition=True, chunking=True):
        """Advance the source circuit and optional local plasticity by one time step."""
        p = self.p
        dt = p.dt
        # 2011 eq13: top-down modulation requires prior bottom-up activation.
        firing = (
            positive_sigmoid(self.c - p.feedback_threshold) * self.mask
            if chunking
            else np.zeros(len(self.c))
        )
        td = p.feedback * (firing @ self.M) if feedback else np.zeros(p.n_items)
        exc_y = p.input_gain * stimulus * self.zi + p.wm_recurrence * self.x + (self.y > 0) * td
        exc_x = p.input_gain * stimulus * self.zi + p.x_recurrence * self.y
        inh_x = 1.25 * (exc_x.sum() - exc_x)
        newy = shunt(self.y, exc_y, 0, dt)
        newx = shunt(self.x, exc_x, inh_x, dt)
        if chunking:
            excitation = p.field_gain / self.size * (
                self.W @ (self.x * self.zw)
            ) + self.size * positive_sigmoid(self.c * self.zc)
            inhibition = (
                p.inhibition * self.masking(positive_sigmoid(self.c, 0.2) * self.mask)
                if competition
                else np.zeros(len(self.c))
            )
            newc = shunt(self.c, excitation, inhibition, dt, lower=1.0, speed=2.0)
            newc = np.where(self.mask > 0, newc, -1.0)
        else:
            newc = np.zeros(len(self.c))
        if learn and chunking:
            # 2014 eq13 on allowed anatomical supports; independent outstar memory.
            activity = self.x[None, :] * self.support
            total = activity.sum(axis=1)
            target = np.divide(
                activity, total[:, None], out=np.zeros_like(activity), where=total[:, None] > 1e-14
            )
            gate = positive_sigmoid(self.c, 0.75) * self.mask
            alpha = 1 - np.exp(-p.learning_rate * gate * total * dt)
            self.W += alpha[:, None] * (target - self.W)
            self.M += alpha[:, None] * (target - self.M)
        self.zi = habituate(self.zi, stimulus, dt)
        self.zw = habituate(self.zw, self.x, dt)
        self.zc = habituate(self.zc, np.maximum(self.c, 0), dt)
        self.x, self.y, self.c = newx, newy, newc
        if not all(np.isfinite(a).all() for a in (self.x, self.y, self.c, self.W, self.M)):
            raise FloatingPointError("Nonfinite state")

    def state(self):
        """Concatenate working memory and positive, unlesioned chunk activity."""
        return np.concatenate([self.x, self.y, np.maximum(self.c, 0) * self.mask])

    def run(self, stream, learn=False, **ablation):
        """Reset trial activity and process the supplied temporal sensory stream."""
        self.reset()
        trace = []
        for stimulus in stream:
            self.step(stimulus, learn=learn, **ablation)
            trace.append(self.state())
        return np.array(trace)

    def save(self, path):
        """Save learned filters and the circuit parameters for later inspection."""
        np.savez_compressed(
            path,
            W=self.W,
            M=self.M,
            support=self.support,
            size=self.size,
            params=str(asdict(self.p)),
        )
