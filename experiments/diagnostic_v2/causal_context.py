"""Fixed past-only aggregation, with explicit resets at known trial boundaries."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque


class CausalVoteBuffer:
    """Online form of the registered five-window rule, reset only at known boundaries."""
    def __init__(self):
        self.history=deque();self.counts=np.zeros(8,dtype=np.int64)

    def reset(self):
        self.history.clear();self.counts.fill(0)

    def update(self,value):
        value=int(value)
        if not 0<=value<8:raise ValueError('Expected class ID 0..7')
        if len(self.history)==5:self.counts[self.history.popleft()]-=1
        self.history.append(value);self.counts[value]+=1
        maximum=self.counts.max()
        return value if self.counts[value]==maximum else int(np.flatnonzero(self.counts==maximum)[0])


def aggregate(prediction,history=5):
    x=np.asarray(prediction)
    if x.ndim!=2 or history<1 or np.any((x<0)|(x>7)):raise ValueError('Expected trial by window class IDs')
    out=np.empty_like(x)
    for i in range(len(x)):
        for j in range(x.shape[1]):
            counts=np.bincount(x[i,max(0,j-history+1):j+1],minlength=8)
            tied=np.flatnonzero(counts==counts.max())
            out[i,j]=x[i,j] if x[i,j] in tied else tied[0]
    return out


def summarize(stem,fold,z,rows,positions,phase):
    ids=np.unique(z['trial']);n=len(z['trial'])//len(ids)
    pred=np.stack([z['prediction'][z['trial']==i] for i in ids])
    truth=np.stack([z['truth'][z['trial']==i] for i in ids])
    out=aggregate(pred);assert np.array_equal(out[:,0],pred[:,0])
    before=pred==truth;after=out==truth
    rows.append(dict(model=stem,phase=phase,fold=fold,windows=pred.size,windows_per_trial=n,history_windows=5,
        base_accuracy=float(before.mean()),causal_accuracy=float(after.mean()),corrected=int((~before&after).sum()),
        harmed=int((before&~after).sum()),first_window_unchanged=True))
    for j in range(n):positions.append(dict(model=stem,phase=phase,fold=fold,start_seconds=1.5+j*.1,
        windows=len(ids),base_accuracy=float(before[:,j].mean()),causal_accuracy=float(after[:,j].mean())))


if __name__=='__main__':
    root=RUN_ROOT;p=root/'private';r=root/'results/diagnostic_v2'
    split=json.loads((p/'splits.json').read_text());dev=np.array(split['train']);held=set(split['test'])
    rows=[];positions=[]
    for name in ['XGB','RF','XGB_win300']:
        for k in range(5):
            z=np.load(p/f'development/{name}_{k}_predictions.npz',allow_pickle=False)
            assert not set(dev[np.unique(z['trial'])])&held
            summarize(name,k,z,rows,positions,'original_development')
    outer=r/'outer.csv'
    if outer.exists():
        for q in pd.read_csv(outer).itertuples():
            z=np.load(p/f'diagnostic_v2/outer_{q.outer_fold}_-1_{q.candidate}.npz',allow_pickle=False)
            assert not set(z['trial'])&held
            summarize(q.candidate,q.outer_fold,z,rows,positions,'nested_outer')
    pd.DataFrame(rows).to_csv(r/'causal_context.csv',index=False)
    pd.DataFrame(positions).to_csv(r/'causal_context_positions.csv',index=False)
    print(pd.DataFrame(rows).groupby(['phase','model'])[['base_accuracy','causal_accuracy','corrected','harmed']].mean().to_string())
