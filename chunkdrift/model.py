"""One-pass feedback autoencoder; exact gradients, no output constraints."""
from copy import deepcopy
import numpy as np


def softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Model:
    def __init__(self, kind, seed, hidden=48, chunks=24, gain=.5, temperature=.5):
        if kind not in {"flat", "dense", "competitive"}:
            raise ValueError(kind)
        self.kind, self.gain, self.temperature = kind, gain, temperature
        self.lower_gain = 1 - gain
        rng = np.random.default_rng(seed)
        self.w = {"E": rng.normal(0, 1 / np.sqrt(36), (36, hidden)),
                  "D": rng.normal(0, 1 / np.sqrt(hidden), (hidden, 36))}
        if kind != "flat":
            self.w["P"] = rng.normal(0, 1 / np.sqrt(hidden), (chunks, hidden))
            self.w["B"] = rng.normal(0, 1 / np.sqrt(chunks), (chunks, hidden))
        self.lower_mask = np.ones(hidden)
        self.chunk_mask = np.ones(chunks)

    def copy(self):
        return deepcopy(self)

    @property
    def parameter_count(self):
        return sum(a.size for a in self.w.values())

    def forward(self, x, decoder=None):
        raw = np.tanh(x @ self.w["E"])
        u = raw * self.lower_mask
        if self.kind == "flat":
            base = c = np.zeros((len(x), 0))
            h = u
        else:
            scores = u @ self.w["P"].T / self.temperature
            base = (softmax(scores) if self.kind == "competitive" else
                    np.tanh(scores) / np.sqrt(len(self.chunk_mask)))
            c = base * self.chunk_mask
            # The gain=0 intervention removes feedback without rescaling u.
            h = (self.lower_gain * u + self.gain * (c @ self.w["B"])) * self.lower_mask
        logits = (h @ (self.w["D"] if decoder is None else decoder)).reshape(-1, 4, 9)
        return {"raw": raw, "u": u, "base": base, "c": c, "h": h,
                "logits": logits, "prob": softmax(logits)}

    def loss_grad(self, x, target):
        a = self.forward(x)
        labels = np.eye(9)[target]
        loss = -np.log(a["prob"][np.arange(len(x))[:, None],
                                  np.arange(4), target] + 1e-300).mean()
        dl = ((a["prob"] - labels) / (len(x) * 4)).reshape(-1, 36)
        grad = {"D": a["h"].T @ dl}
        dh = dl @ self.w["D"].T
        if self.kind == "flat":
            du = dh
        else:
            dh = dh * self.lower_mask
            grad["B"] = self.gain * a["c"].T @ dh
            dc = self.gain * (dh @ self.w["B"].T) * self.chunk_mask
            base = a["base"]
            if self.kind == "competitive":
                ds = base * (dc - (dc * base).sum(axis=1, keepdims=True))
            else:
                k = len(self.chunk_mask)
                ds = dc * (1 - (base * np.sqrt(k)) ** 2) / np.sqrt(k)
            ds /= self.temperature
            grad["P"] = ds.T @ a["u"]
            du = self.lower_gain * dh + ds @ self.w["P"]
        grad["E"] = x.T @ (du * self.lower_mask * (1 - a["raw"] ** 2))
        return float(loss), grad

    def update(self, x, target, rate):
        loss, grad = self.loss_grad(x, target)
        for name in self.w:
            self.w[name] -= rate * grad[name]
            if not np.isfinite(self.w[name]).all():
                raise FloatingPointError(name)
        return loss

    def noise(self, seed, step, level, scales):
        for name, a in self.w.items():
            # Independent array streams: adding/removing arrays cannot alter E,D noise.
            key = {"E": 1, "D": 2, "P": 3, "B": 4}[name]
            rng = np.random.default_rng([seed, step, key, 911])
            a += level * scales[name] * rng.normal(size=a.shape)

    def intervene(self, intervention, seed, fraction=.25):
        rng = np.random.default_rng([seed, 381])
        if intervention == "feedback_off":
            self.gain = 0.0
        elif intervention == "chunk_lesion":
            idx = rng.choice(len(self.chunk_mask), round(fraction * len(self.chunk_mask)), False)
            self.chunk_mask[idx] = 0
        elif intervention == "lower_lesion":
            idx = rng.choice(len(self.lower_mask), round(fraction * len(self.lower_mask)), False)
            self.lower_mask[idx] = 0
        else:
            raise ValueError(intervention)
