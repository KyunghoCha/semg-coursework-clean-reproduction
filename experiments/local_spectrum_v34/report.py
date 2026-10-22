"""Aggregate reporting after verification, never selection from outer results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import R,P,inputs,source800,decode,base,digest


def main():
    assert (R/'verification.json').exists() and (R/'independent_features.json').exists()
    _,y,_,schedule=inputs();rows=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19)
        fixed=decode(np.load(source800('inner',k))['probability']).argmax(1)
        incumbent=base('inner',k,ids,y).argmax(1)
        new=decode(np.load(P/f'inner_local80_800_{k}.npz')['probability']).argmax(1)
        rows.append(dict(fold=k,windows=len(truth),fixed800_correct=int((fixed==truth).sum()),
                         incumbent_v33_correct=int((incumbent==truth).sum()),
                         local80_correct=int((new==truth).sum()),
                         change_vs_fixed800=int((new==truth).sum()-(fixed==truth).sum()),
                         change_vs_incumbent=int((new==truth).sum()-(incumbent==truth).sum()),
                         fixes_incumbent_errors=int(((new==truth)&(incumbent!=truth)).sum()),
                         loses_incumbent_correct=int(((new!=truth)&(incumbent==truth)).sum())))
    pd.DataFrame(rows).to_csv(R/'inner_comparison.csv',index=False)
    fits=pd.read_csv(R/'fits.csv');pool=pd.read_csv(R/'pooled.csv');outer=pd.read_csv(R/'outer.csv')
    choices=json.loads((R/'selection.json').read_text())['use_new_model']
    inner_summary=[]
    for name,col in [('fixed_410_XGB800','fixed800_correct'),('incumbent_v33','incumbent_v33_correct'),('local80_490_XGB800','local80_correct')]:
        vals=np.array([row[col]/row['windows'] for row in rows])
        inner_summary.append(dict(model=name,correct=sum(row[col] for row in rows),windows=14592,mean_accuracy=float(vals.mean()),fold_sd=float(vals.std())))
    paired=[]
    for k in range(3):
        old=np.load(P/f'reference_{k}.npz');new=np.load(P/f'selected_{k}.npz')
        np.testing.assert_array_equal(old['trial'],new['trial']);np.testing.assert_array_equal(old['truth'],new['truth'])
        truth=old['truth'];a=old['prediction']==truth;b=new['prediction']==truth
        trial_change=b.reshape(-1,19).sum(1)-a.reshape(-1,19).sum(1)
        paired.append(dict(fold=k,fixed_errors=int((~a&b).sum()),lost_correct=int((a&~b).sum()),
                           improved_trials=int((trial_change>0).sum()),worse_trials=int((trial_change<0).sum()),
                           unchanged_trials=int((trial_change==0).sum())))
    pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False)
    report=dict(input_ms=200,windows_per_trial=19,features_candidate=490,single_estimator=True,
                use_new_model=choices,inner_summary=inner_summary,outer=pool.to_dict('records'),paired_changes=paired,
                new_fit_count=len(fits),fit_seconds=float(fits.fit_seconds.sum()),
                model_bytes_min=int(fits.model_bytes.min()),model_bytes_max=int(fits.model_bytes.max()),
                changed_outer_conditions=False,independent_external_test=False,
                selection_sha256=digest(R/'selection.json'))
    with (R/'summary.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    a=np.arange(3)
    for dx,col,label,color in [(-.25,'fixed800_correct','410D, XGB800','#4c78a8'),(0,'incumbent_v33_correct','v33 incumbent','#999999'),(.25,'local80_correct','490D, XGB800','#e08c35')]:
        ax[0].plot(a,[row[col]/row['windows']*100 for row in rows],marker='o',label=label,color=color)
    ax[0].set(xticks=a,xticklabels=['Fold 0','Fold 1','Fold 2'],ylim=(80,95),ylabel='Window accuracy (%)',title='Inner evaluation: 4,864 windows / fold')
    ax[0].legend(fontsize=8,loc='lower right')
    for role,marker,label in [('reference','o','v33 incumbent'),('selected','x','v34 selection procedure')]:
        d=outer[outer.role==role].sort_values('fold');ax[1].plot(a,d.accuracy*100,marker=marker,label=label)
    ax[1].set(xticks=a,xticklabels=['Fold 0','Fold 1','Fold 2'],ylim=(88,94),ylabel='Window accuracy (%)',title='Outer development: 9,728 windows / fold')
    ax[1].legend(fontsize=8)
    for item in ax:item.grid(axis='y',alpha=.2);item.set_axisbelow(True)
    fig.savefig(R/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
