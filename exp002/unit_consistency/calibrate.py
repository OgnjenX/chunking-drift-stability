"""Recheck acquisition/calibration after explicit physical-unit conversion.

Uses the identical development training streams and saved ordinary GRU from
v1: no extra training or outcome-selected budget is given to either arm.
"""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .model import PhysicalUnitsField
from ..model import Params
from ..controls import Readout,GRU,scores
from ..task import stream,trials
from ..develop import LEVELS
from ..develop_probe import DEV_OLD,DEV_LONG


def main():
    out=Path('exp002/results/unit-consistency-calibration');out.mkdir(exist_ok=False,parents=True)
    p=Params(pulse=.75,feedback=8.)
    records=[];codes=[]
    for seed in [101,102,103]:
        field=PhysicalUnitsField(seed,p);head=Readout(len(field.state()))
        frozen=PhysicalUnitsField(seed,p);fixed_head=Readout(len(frozen.state()))
        before=field.W.copy()
        for phase,seqs in [(1,DEV_OLD),(2,DEV_LONG)]:
            for seq,a in trials(seqs,p,seed,30,{'delay':2.,'noise':.005},phase):
                head.update(field.run(a,learn=True)[-1],seq)
                fixed_head.update(frozen.run(a,learn=False)[-1],seq)
        old=np.load(Path('exp002/results/development-calibration')/f's{seed}_readouts.npz')
        gru=GRU(seed,hidden=21);gru.w={k:old[k].copy() for k in gru.w}
        levels=LEVELS+[{'name':'training_matched','delay':2.,'noise':.005,'missing':0,'interference':0.}]
        for level in levels:
            clean=[];unlearned=[]
            for rep in range(4):
                for i,seq in enumerate(DEV_OLD+DEV_LONG):
                    args={k:v for k,v in level.items() if k!='name'}
                    a=stream(seq,p,np.random.default_rng([seed,95,rep,i]),**args)
                    state=field.run(a)[-1];base=frozen.run(a)[-1]
                    for label,logits in [('field',head.predict(state)),('frozen_filters',fixed_head.predict(base)),('gru',gru.forward(a)[0])]:
                        records.append({'seed':seed,'condition':level['name'],'sequence':seq,'rep':rep,'model':label,**scores(logits,seq)})
                    if level['name']=='training_matched' and rep==0:
                        clean.append(state);unlearned.append(base)
                        for intervention,opts in [('feedback_off',{'feedback':False}),('competition_off',{'competition':False}),('chunks_off',{'chunking':False})]:
                            h=field.run(a,**opts)[-1]
                            records.append({'seed':seed,'condition':level['name'],'sequence':seq,'rep':rep,'model':intervention,**scores(head.predict(h),seq)})
            if level['name']=='training_matched':
                c=np.array(clean)[:,8:];b=np.array(unlearned)[:,8:]
                codes.append({'seed':seed,'trained_winners':int(len(set(c.argmax(axis=1)))),
                              'untrained_filter_winners':int(len(set(b.argmax(axis=1)))),
                              'trained_mean_active':float((c>.2).sum(axis=1).mean()),
                              'untrained_mean_active':float((b>.2).sum(axis=1).mean()),
                              'filter_relative_change':float(np.linalg.norm(field.W-before)/np.linalg.norm(before))})
                np.savez_compressed(out/f's{seed}_codes.npz',learned=np.array(clean),untrained=np.array(unlearned),W=field.W,M=field.M,head=head.B,frozen_head=fixed_head.B)
        (out/'trials.json').write_text(json.dumps(records,indent=2)+'\n')
        (out/'mechanisms.json').write_text(json.dumps(codes,indent=2)+'\n')
        print(codes[-1],flush=True)
    summarized=[]
    for level in LEVELS+[{'name':'training_matched'}]:
        means={model:float(np.mean([r['exact'] for r in records if r['condition']==level['name'] and r['model']==model])) for model in ['field','frozen_filters','gru']}
        summarized.append({'condition':level['name'],**means})
    decision={'parameters':asdict(p),'development_only':True,'seeds':[101,102,103],
              'same_600_training_trials_as_v1':True,'reused_gru_weights_from_identical_training':True,
              'performance':summarized,
              'shared_headroom_calibrated':any(.45<=r['field']<=.85 and .45<=r['gru']<=.85 for r in summarized if r['condition'] not in ['clean','training_matched']),
              'confirmation_seeds_executed':[]}
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
