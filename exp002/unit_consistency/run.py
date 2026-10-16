"""Bounded source-unit check, development data only."""
import json
from pathlib import Path
import numpy as np
from .model import PhysicalUnitsField
from ..model import Params
from ..develop_probe import DEV_OLD,DEV_LONG
from ..task import stream


def main():
    out=Path('exp002/results/unit-consistency-development')
    out.mkdir(exist_ok=False,parents=True)
    rows=[]
    for seed in [101,102,103]:
      for pulse in [.75,2.,6.]:
        for feedback in [1.,8.]:
          p=Params(pulse=pulse,feedback=feedback)
          m=PhysicalUnitsField(seed,p);before=m.W.copy()
          for rep in range(8):
            for i,seq in enumerate(DEV_OLD+DEV_LONG):
              m.run(stream(seq,p,np.random.default_rng([seed,rep,i]),delay=2.),learn=True)
          states=np.array([m.run(stream(seq,p,np.random.default_rng(103),delay=2.))[-1] for seq in DEV_OLD+DEV_LONG])
          c=states[:,8:];winners=c.argmax(axis=1)
          comparisons=[]
          for i,a in enumerate(DEV_LONG):
            for j,b in enumerate(DEV_LONG[:i]):
              if set(a)==set(b):comparisons.append(float(winners[10+i]!=winners[10+j]))
          row={'seed':seed,'pulse':pulse,'feedback':feedback,'winner_count':int(len(set(winners))),
               'winner_sizes':m.size[winners].tolist(),
               'same_set_order_winner_discrimination':float(np.mean(comparisons)),
               'mean_active_over_threshold':float((c>.2).sum(axis=1).mean()),
               'weight_relative_change':float(np.linalg.norm(m.W-before)/np.linalg.norm(before))}
          rows.append(row)
          np.savez_compressed(out/f's{seed}_p{pulse}_f{feedback}.npz',states=states,W=m.W,M=m.M,before=before)
          (out/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
          print({k:v for k,v in row.items() if k!='winner_sizes'},flush=True)


if __name__=='__main__':main()
