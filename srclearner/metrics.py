import numpy as np


def accuracy(model, trials):
    return float(np.mean([model.predict(t.sequence) == t.outcome for t in trials]))


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    numerator = np.linalg.norm(a.T @ b) ** 2
    denominator = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(numerator / denominator) if denominator else None


def paired_interval(a, b):
    """Descriptive development interval; bootstrap over independent worlds."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(480)
    draws = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    return dict(
        mean=float(d.mean()),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
        n_worlds=len(d),
    )
