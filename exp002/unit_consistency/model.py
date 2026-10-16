"""Explicit adapter between normalized 2014 filters and 2011 physical currents."""

import numpy as np

from ..model import TemporalField, positive_sigmoid


class PhysicalUnitsField(TemporalField):
    """Convert normalized learned filters into physical source current units."""

    def step(self, stimulus, learn=False, feedback=True, competition=True, chunking=True):
        """Integrate physical currents, then update normalized memories locally."""
        x, c = self.x.copy(), self.c.copy()
        w, m = self.W, self.M
        scale = 0.1 * self.size[:, None]
        # Physical current conversion, not weight normalization after an output test.
        self.W = w * scale
        self.M = m * scale
        try:
            super().step(
                stimulus, learn=False, feedback=feedback, competition=competition, chunking=chunking
            )
        finally:
            self.W, self.M = w, m
        if learn and chunking:
            activity = x[None, :] * self.support
            total = activity.sum(axis=1)
            target = np.divide(
                activity, total[:, None], out=np.zeros_like(activity), where=total[:, None] > 1e-14
            )
            gate = positive_sigmoid(c, 0.75) * self.mask
            alpha = 1 - np.exp(-self.p.learning_rate * gate * total * self.p.dt)
            self.W += alpha[:, None] * (target - self.W)
            self.M += alpha[:, None] * (target - self.M)
