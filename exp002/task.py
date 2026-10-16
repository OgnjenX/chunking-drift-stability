"""Sequential stimulus streams, variable lengths, delays and matched interference."""
from itertools import permutations
import numpy as np

# Recurrent subsequences AB, BC, CD appear both alone and in larger lists.
OLD=['A','B','C','D','AB','BA','BC','CB','CD','DC']
LONG=['ABC','ACB','BCD','BDC','ABCD','ABDC','BACD','BCAD','CDAB','DCBA']
ALL=OLD+LONG
UNSEEN=[''.join(x) for k in (3,4) for x in permutations('ABCD',k) if ''.join(x) not in ALL]


def stream(seq, params, rng, delay=2., noise=0., missing=0, interference=0):
    n=params.n_items
    vectors=[]
    pulse_steps=round(params.pulse/params.dt)
    gap_steps=round(params.gap/params.dt)
    missing_positions=set(rng.choice(len(seq),min(missing,len(seq)),False).tolist()) if missing else set()
    for j,symbol in enumerate(seq):
        v=np.eye(n)['ABCD'.index(symbol)] if j not in missing_positions else np.ones(n)/n
        vectors.extend([v.copy() for _ in range(pulse_steps)])
        vectors.extend([np.zeros(n) for _ in range(gap_steps)])
    for t in range(round(delay/params.dt)):
        v=np.zeros(n)
        # Distractor items occupy brief intervals during the retention delay.
        if interference and t % max(1,pulse_steps+gap_steps) < max(1,pulse_steps//4):
            # Trial-local draws, identical for all architectures/ablations.
            v=np.eye(n)[rng.integers(n)]*interference
        vectors.append(v)
    a=np.array(vectors)
    if noise:
        a=np.maximum(a+rng.normal(0,noise,size=a.shape),0)
    return a


def trials(sequences, params, seed, repetitions, difficulty, phase):
    for rep in range(repetitions):
        rng=np.random.default_rng([seed,phase,rep,9102])
        for seq_index in rng.permutation(len(sequences)):
            seq=sequences[seq_index]
            trial_seed=[seed,phase,rep,int(seq_index),103]
            yield seq,stream(seq,params,np.random.default_rng(trial_seed),**difficulty)
