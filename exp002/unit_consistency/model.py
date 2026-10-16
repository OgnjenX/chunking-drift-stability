"""Explicit adapter between normalized 2014 filters and 2011 physical currents."""
import numpy as np
from ..model import TemporalField,positive_sigmoid


class PhysicalUnitsField(TemporalField):
    def step(self,stimulus,learn=False,**kwargs):
        x,c=self.x.copy(),self.c.copy()
        w,m=self.W,self.M
        scale=.1*self.size[:,None]
        # Physical current conversion, not weight normalization after an output test.
        self.W=w*scale;self.M=m*scale
        try:
            super().step(stimulus,learn=False,**kwargs)
        finally:
            self.W,self.M=w,m
        if learn and kwargs.get('chunking',True):
            activity=x[None,:]*self.support
            total=activity.sum(axis=1)
            target=np.divide(activity,total[:,None],out=np.zeros_like(activity),where=total[:,None]>1e-14)
            gate=positive_sigmoid(c,.75)*self.mask
            alpha=1-np.exp(-self.p.learning_rate*gate*total*self.p.dt)
            self.W+=alpha[:,None]*(target-self.W)
            self.M+=alpha[:,None]*(target-self.M)
