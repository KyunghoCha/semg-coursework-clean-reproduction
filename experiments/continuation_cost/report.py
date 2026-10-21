"""Aggregate verified stages without changing model selection or private artifacts."""
import hashlib
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/continuation_report'
if (R/'complete.json').exists() and '--render-only' not in sys.argv:raise FileExistsError('Preserve completed report')
for name in ['nested_selection_v10','line_projection_v11','probability_blend_v12','forest_transfer_v13']:
    assert (RUN_ROOT/f'results/{name}/verification.json').exists()
R.mkdir(exist_ok=True)
source=[]
def table(stage,file):
    path=RUN_ROOT/f'results/{stage}/{file}'
    source.append(dict(path=str(path.relative_to(RUN_ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.read_csv(path)
v10=table('nested_selection_v10','outer_by_fold.csv')
v12=table('probability_blend_v12','outer.csv')
v13=table('forest_transfer_v13','outer.csv')
series=[('Fixed fine64',v10[v10.role=='reference']),
        ('v10 inner selection',v10[v10.role=='selected']),
        ('Fixed narrow64',v12[v12.role=='narrow']),
        ('Fixed 50:50 blend',v12[v12.role=='fixed_half']),
        ('v12 inner-selected blend',v12[v12.role=='selected']),
        ('v13 inner selection',v13[v13.role=='selected'])]
fig,ax=plt.subplots(figsize=(9.5,4.8),layout='constrained')
label_x=max(frame.accuracy.max()*100 for _,frame in series)+.14
for i,(label,rows) in enumerate(series):
    a=rows.sort_values('fold').accuracy.to_numpy()*100
    ax.errorbar(a.mean(),i,xerr=a.std(ddof=0),fmt='o',capsize=4)
    ax.scatter(a,np.full(3,i),s=18,color='gray',alpha=.6)
    ax.text(label_x,i,f'{a.mean():.3f}%',va='center',fontsize=9)
ax.set(yticks=range(len(series)),yticklabels=[a[0] for a in series],xlabel='200 ms window accuracy (%)',
       title='Same outer folds; all 29,184 windows retained')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
left,right=ax.get_xlim();ax.set_xlim(left,label_x+.42)
fig.savefig(R/'selection_comparison.png',dpi=170);plt.close(fig)
rows=[]
for label,frame in series:
    a=frame.accuracy.to_numpy()
    rows.append(dict(strategy=label,accuracy=float(a.mean()),fold_std=float(a.std(ddof=0)),folds=3,windows=29184))
pd.DataFrame(rows).to_csv(R/'selection_summary.csv',index=False)
# Fixed signal-projection ablation has different folds; keep in its own figure/table.
base=table('fine_spectrum_v8','scores.csv');removed=table('line_projection_v11','scores.csv')
a=base.sort_values('fold').accuracy.to_numpy()*100;b=removed.sort_values('fold').accuracy.to_numpy()*100
assert len(a)==len(b)==3
fig,ax=plt.subplots(figsize=(6.7,4.6),layout='constrained')
for fold in range(3):ax.plot([0,1],[a[fold],b[fold]],'o-',label=f'Fold {fold}')
ax.set(xticks=[0,1],xticklabels=['Original window','300/420 Hz projection removed'],ylabel='Window accuracy (%)',title='Fixed model, unchanged 200 ms / 19 windows')
ax.grid(alpha=.2);ax.legend();fig.savefig(R/'component_removal.png',dpi=170);plt.close(fig)
pd.DataFrame(dict(fold=range(3),original_accuracy=a/100,removed_accuracy=b/100,difference_pp=b-a)).to_csv(R/'projection_paired.csv',index=False)
forests=table('forest_transfer_v13','inner_candidates.csv')
order=['fine64','narrow64','RF100_fine','ET400_fine']
fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for j,column in enumerate(['accuracy','model_bytes']):
    for i,name in enumerate(order):
        values=forests[forests.model==name][column].to_numpy()*(100 if j==0 else 1e-6)
        axes[j].bar(i,values.mean(),yerr=values.std(ddof=0),capsize=4,color=['#3377aa','#ee9944','#44aa88','#aa66aa'][i])
        axes[j].scatter(np.repeat(i,len(values)),values,color='black',s=12)
    axes[j].set(xticks=range(4),xticklabels=['Fine XGB','Narrow XGB','RF100','ET400'])
    axes[j].grid(axis='y',alpha=.2)
axes[0].set(ylabel='Inner selection accuracy (%)',title='Same 768 training / 256 selection trials')
axes[1].set(ylabel='Stored model + scaler (MB)',title='Actual size; different tree structures')
fig.savefig(R/'forest_inner_comparison.png',dpi=170);plt.close(fig)
# Error locations use the selected blend's already-existing OOF predictions.
profile=[];classes=[];paired=[]
for fold in range(3):
    z=np.load(RUN_ROOT/f'private/probability_blend_v12/selected_{fold}.npz')
    ref=np.load(RUN_ROOT/f'private/probability_blend_v12/fine_{fold}.npz')
    np.testing.assert_array_equal(z['truth'],ref['truth']);np.testing.assert_array_equal(z['trial'],ref['trial'])
    for name,q in [('selected_blend',z),('fine_reference',ref)]:
        truth=q['truth'];pred=q['prediction'];ok=truth==pred
        for i in range(19):profile.append(dict(strategy=name,fold=fold,start_seconds=1.5+.1*i,correct=int(ok[i::19].sum()),windows=len(ok[i::19]),accuracy=float(ok[i::19].mean())))
        for c in range(8):
            mask=truth==c;classes.append(dict(strategy=name,fold=fold,label=c,correct=int(ok[mask].sum()),windows=int(mask.sum()),accuracy=float(ok[mask].mean())))
    a=(z['prediction']==z['truth']).reshape(-1,19);b=(ref['prediction']==ref['truth']).reshape(-1,19)
    diff=a.sum(axis=1)-b.sum(axis=1)
    paired.append(dict(fold=fold,only_blend_correct=int((a&~b).sum()),only_reference_correct=int((b&~a).sum()),both_wrong=int((~a&~b).sum()),trials_improved=int((diff>0).sum()),trials_worse=int((diff<0).sum()),trials_tied=int((diff==0).sum())))
pd.DataFrame(profile).to_csv(R/'blend_positions.csv',index=False);pd.DataFrame(classes).to_csv(R/'blend_class_support.csv',index=False);pd.DataFrame(paired).to_csv(R/'blend_paired.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
for name,label in [('fine_reference','Fine reference'),('selected_blend','Inner-selected blend')]:
    frame=pd.DataFrame(profile);values=frame[frame.strategy==name].groupby('start_seconds')[['correct','windows']].sum()
    ax.plot(values.index,100*values.correct/values.windows,'o-',label=label)
ax.set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',title='Same OOF trials and current-window information');ax.grid(alpha=.2);ax.legend()
fig.savefig(R/'blend_positions.png',dpi=170);plt.close(fig)
(R/'complete.json').write_text(json.dumps(dict(source_tables=source,external_independent_data=False,new_fits=0,selection_changed=False),indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(paired).sum(numeric_only=True).to_string())
