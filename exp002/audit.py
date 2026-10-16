"""Reproduce source learning without the GRU or confirmatory seed exposure."""
import json,subprocess,hashlib,platform,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .model import Params,TemporalField
from .controls import Readout,scores
from .evaluate import serialize
from .task import trials,stream
from .develop_probe import DEV_OLD,DEV_LONG


def main():
    root=Path('exp002/results/development-calibration')
    p=Params(**json.loads((root/'decision.json').read_text())['parameters'])
    rows=[];audits=[]
    for seed in [101,102,103]:
        m=TemporalField(seed,p);head=Readout(len(m.state()))
        for phase,seqs in [(1,DEV_OLD),(2,DEV_LONG)]:
            for seq,a in trials(seqs,p,seed,30,{'delay':2.,'noise':.005},phase):
                h=m.run(a,learn=True)[-1];head.update(h,seq)
            # New measurement with training-matched distribution, not the earlier
            # noise-free .5-delay distribution-shift diagnostic.
            for rep in range(4):
                for i,seq in enumerate(DEV_OLD+DEV_LONG):
                    a=stream(seq,p,np.random.default_rng([seed,930,rep,i]),delay=2.,noise=.005)
                    h=m.run(a)[-1]
                    rows.append({'seed':seed,'phase':phase,'rep':rep,'sequence':seq,**scores(head.predict(h),seq)})
        saved=np.load(root/f's{seed}_field.npz');readout=np.load(root/f's{seed}_readouts.npz')
        for key in ('W','M'):np.testing.assert_array_equal(getattr(m,key),saved[key])
        np.testing.assert_array_equal(head.B,readout['field'])
        audits.append({'seed':seed,'field_filters_and_task_readout_bitwise_reproduced':True})
    # Every tracked EXP001 byte must match its archived commit, including README.
    base='af5272bf88b569268a3a993264a7bba16c38645f'
    tree=subprocess.check_output(['git','ls-tree','-rz','--full-tree',base]).split(b'\0')
    files=0
    for entry in tree:
        if not entry:continue
        meta,path=entry.split(b'\t',1);expected=meta.split()[2].decode();path=Path(path.decode())
        actual=subprocess.check_output(['git','hash-object',str(path)],text=True).strip()
        assert actual==expected,str(path)
        files+=1
    result={'EXP001_base_commit':base,'EXP001_files_verified_byte_identical':files,
            'source_replays':audits,'development_only':True,'confirmation_seeds_executed':[],
            'retrospective_environment':{'recorded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                         'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
            'source_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path('exp002').glob('*.py')}}
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'training-matched-retention.json').write_text(json.dumps(rows,indent=2,default=serialize)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','retrospective_environment')},indent=2))


if __name__=='__main__':main()
