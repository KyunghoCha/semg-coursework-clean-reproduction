"""Aggregate-only figures and error counts for the verified fixed comparison."""
from semg.paths import RUN_ROOT
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=RUN_ROOT/'results/latent_subclass_v5';P=RUN_ROOT/'private/latent_subclass_v5'
assert json.loads((R/'verification.json').read_text())['models_restored']==6
rows=pd.read_csv(R/'scores.csv');positions=pd.read_csv(R/'positions.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
labels={'global8':'Direct 8-class target','subclass64':'64 training targets, 8 outputs'}
for name,g in rows.groupby('model'):
    axes[0].plot(g.fold+1,g.accuracy*100,'o-',label=labels[name])
    pos=positions[positions.model==name].groupby('start_seconds').accuracy.mean()
    axes[1].plot(pos.index,pos*100,'o-',label=labels[name])
axes[0].set(xlabel='Development fold',ylabel='Window accuracy (%)',xticks=[1,2,3],ylim=(65,100),title='Fixed 200 ms input, 19 windows')
axes[1].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',ylim=(65,100),title='All evaluation positions retained')
axes[0].legend(fontsize=8);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
errors=[]
for name in labels:
    z=[np.load(P/f'{name}_{k}.npz',allow_pickle=False) for k in range(3)]
    t=np.concatenate([a['truth'] for a in z]);p=np.concatenate([a['prediction'] for a in z])
    angle=t//4!=p//4;gesture=t%4!=p%4
    errors.append(dict(model=name,angle_only=int((angle&~gesture).sum()),gesture_only=int((~angle&gesture).sum()),both=int((angle&gesture).sum()),errors=int((t!=p).sum())))
pd.DataFrame(errors).to_csv(R/'error_categories.csv',index=False)
