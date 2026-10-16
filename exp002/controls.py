"""Nonchunking recurrent controls and shared sensory sequence readout targets."""

import numpy as np

from .model import store2_rhs


def target(seq):
    """Encode the ordered sequence and padding as a flat supervised target."""
    ids = ["ABCD".index(s) for s in seq] + [4] * (4 - len(seq))
    return np.eye(5)[ids].reshape(20)


def scores(logits, seq):
    """Measure exact sequence, item token, and sequence-length accuracy."""
    pred = np.asarray(logits).reshape(4, 5).argmax(axis=1)
    truth = np.r_[["ABCD".index(s) for s in seq], [4] * (4 - len(seq))].astype(int)
    return {
        "exact": float(np.array_equal(pred, truth)),
        "token": float(np.mean(pred[: len(seq)] == truth[: len(seq)])),
        "length": float(np.array_equal(pred == 4, truth == 4)),
    }


class Readout:
    """Recursive least squares; stores learned sufficient statistics, not replay."""

    def __init__(self, n, regularization=0.1):
        self.P = np.eye(n + 1) / regularization
        self.B = np.zeros((n + 1, 20))
        self.count = 0

    def features(self, state):
        """Append the readout intercept to the state vector."""
        return np.r_[state, 1.0]

    def update(self, state, seq):
        """Update sufficient statistics with one labeled trial."""
        x = self.features(state)
        px = self.P @ x
        k = px / (1 + x @ px)
        self.B += np.outer(k, target(seq) - x @ self.B)
        self.P -= np.outer(k, px)
        self.count += 1

    def predict(self, state):
        """Return sequence logits from the fixed current readout."""
        return self.features(state) @ self.B


# One public operation mirrors the complete STORE2 reference circuit.
class StoreControl:  # pylint: disable=too-few-public-methods
    """STORE2 working-memory circuit without a learned chunk field."""

    def __init__(self, n=4, dt=0.05, gain=0.01):
        self.n, self.dt, self.gain = n, dt, gain
        self.head = Readout(2 * n)

    def run(self, a):
        """Integrate a sensory stream and return the two working-memory stores."""
        x = np.zeros(self.n)
        y = x.copy()
        for v in a:
            # STORE2 gating assumes one unit pulse at a time. Normalize only when
            # noisy overlapping sensory inputs exceed one, to preserve I^C>=0.
            v = v / max(1, v.sum())
            dx, dy = store2_rhs(x, y, v, self.gain)
            x, y = x + self.dt * dx, y + self.dt * dy
        return np.r_[x, y]


class GRU:
    """Trainable ordinary GRU; equal raw streams and per-trial targets, no chunks."""

    def __init__(self, seed, n=4, hidden=21, lr=0.01):
        rng = np.random.default_rng(seed)
        self.hidden, self.lr = hidden, lr
        self.w = {}
        for gate in ("z", "r", "q"):
            self.w["W" + gate] = rng.normal(0, 1 / np.sqrt(n), (n, hidden))
            self.w["U" + gate] = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden))
            self.w["b" + gate] = np.zeros(hidden)
        self.w["bz"][:] = 2.0  # Long initial retention, not a memoryless comparator.
        self.w["D"] = rng.normal(0, 0.1, (hidden, 20))
        self.w["bd"] = np.zeros(20)
        self.m = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.t = 0

    @property
    def parameter_count(self):
        """Count the trainable recurrent and output parameters."""
        return sum(x.size for x in self.w.values())

    @staticmethod
    def sig(x):
        """Evaluate a numerically bounded logistic gate."""
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, stream, cache=False):
        """Process successive items, optionally retaining the gradient cache."""
        h = np.zeros(self.hidden)
        states = []
        w = self.w
        for x in stream:
            prev = h
            z = self.sig(x @ w["Wz"] + prev @ w["Uz"] + w["bz"])
            r = self.sig(x @ w["Wr"] + prev @ w["Ur"] + w["br"])
            q = np.tanh(x @ w["Wq"] + (r * prev) @ w["Uq"] + w["bq"])
            h = z * prev + (1 - z) * q
            if cache:
                states.append((x, prev, z, r, q))
        return h @ w["D"] + w["bd"], h, states

    def loss_grad(self, a, seq):
        """Compute sequence cross-entropy and backpropagate through the GRU."""
        out, h, states = self.forward(a, cache=True)
        logits = out.reshape(4, 5)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        truth = target(seq).reshape(4, 5)
        loss = float(-(truth * np.log(prob + 1e-300)).sum() / 4)
        dl = ((prob - truth) / 4).reshape(20)
        g = {k: np.zeros_like(v) for k, v in self.w.items()}
        w = self.w
        g["D"] = np.outer(h, dl)
        g["bd"] = dl.copy()
        dh = dl @ w["D"].T
        for x, prev, z, r, q in reversed(states):
            dz = dh * (prev - q) * z * (1 - z)
            dq = dh * (1 - z) * (1 - q * q)
            dr = (dq @ w["Uq"].T) * prev * r * (1 - r)
            for key, pre, delta in [("z", prev, dz), ("r", prev, dr), ("q", r * prev, dq)]:
                g["W" + key] += np.outer(x, delta)
                g["U" + key] += np.outer(pre, delta)
                g["b" + key] += delta
            dh = dh * z + dz @ w["Uz"].T + dr @ w["Ur"].T + (dq @ w["Uq"].T) * r
        return loss, g

    def update(self, a, seq):
        """Apply a clipped Adam update using the current labeled sensory trial."""
        loss, g = self.loss_grad(a, seq)
        norm = np.sqrt(sum((v * v).sum() for v in g.values()))
        scale = min(1, 5 / max(norm, 1e-15))  # Ordinary optimizer clipping, no output gate.
        self.t += 1
        for k in self.w:
            d = g[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * d
            self.v[k] = 0.999 * self.v[k] + 0.001 * d * d
            self.w[k] -= (
                self.lr
                * (self.m[k] / (1 - 0.9**self.t))
                / (np.sqrt(self.v[k] / (1 - 0.999**self.t)) + 1e-8)
            )
        return loss
