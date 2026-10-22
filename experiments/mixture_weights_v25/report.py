"""Report the frozen weight comparison and its measured costs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
from sources import P,R,digest,check_hashes


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    costs=json.loads((R/'costs.json').read_text());raw=json.loads((P/'cost_raw.json').read_text());assert digest(P/'cost_raw.json')==costs['raw_timing_sha256'];check_hashes(costs['source_hashes'])
    for row in costs['rows']:
        v=raw[row['role']];assert len(v)==380
        for key,value in dict(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))).items():assert row[key]==value
    inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');pooled=pd.read_csv(R/'pooled.csv');paired=pd.read_csv(R/'paired_changes.csv');betas=json.loads((R/'selection.json').read_text())['fold_betas'];totals=inner.groupby('beta',as_index=False)[['windows','correct']].sum();totals['accuracy']=totals.correct/totals.windows;totals.to_csv(R/'inner_summary.csv',index=False)
    summary=dict(**json.loads((R/'complete.json').read_text()),fold_betas=betas,inner=totals.to_dict('records'),outer=pooled.to_dict('records'),costs=costs['rows'],raw_cost_statistics_recomputed=True,input_ms=200,temporal_context=False,all_19_windows_evaluated=True,external_independent_test=False)
    for key in ['corrected','broken','net','trial_improved','trial_worsened','trial_tied']:summary[key]=int(paired[key].sum())
    old=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/pooled.csv').query("role=='selected'").iloc[0];summary['v21_correct']=int(old.correct);summary['net_vs_v21']=int(pooled.query("role=='selected'").iloc[0].correct-old.correct)
    label=json.loads((R/'label_cost.json').read_text());timing=np.load(P/'label_cost_raw.npz');assert digest(P/'label_cost_raw.npz')==label['raw_timing_sha256'];check_hashes(label['source_hashes'])
    values=timing['milliseconds'];mask=timing['svm_called'];assert len(values)==9728 and float(mask.mean())==label['call_fraction']
    for key,v in [('all',values),('svm_called',values[mask]),('svm_skipped',values[~mask])]:
        expected=dict(count=len(v))
        if len(v):expected.update(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(np.max(v)))
        assert expected==label[key]
    summary['label_only_cost']={k:v for k,v in label.items() if k!='source_hashes'};summary['label_only_cost_statistics_recomputed']=True
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
    for fold in range(3):
        part=inner[inner.fold==fold];ax.plot(part.beta,100*part.accuracy,'o-',label=f'Inner fold {fold}');row=part[np.isclose(part.beta,betas[fold])].iloc[0];ax.scatter([row.beta],[100*row.accuracy],marker='*',s=150,edgecolors='black',zorder=4)
    ax.set(xlabel='SVM probability weight beta',ylabel='8-class window accuracy (%)',title='Weights selected only on inner observations; current 200 ms');ax.legend();ax.grid(alpha=.2);fig.savefig(R/'inner_weights.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained');prior=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/outer.csv');x=np.arange(3)
    for offset,data,role,label in [(-.25,prior,'selected','Fixed v21'),(0,outer,'incumbent','v24 XGB replacement'),(.25,outer,'selected','v25 inner-selected beta')]:
        vals=data[data.role==role].sort_values('fold').accuracy.to_numpy()*100;ax.bar(x+offset,vals,.24,label=label)
        for a,b in zip(x+offset,vals):ax.text(a,b+.5,f'{b:.2f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Outer 0','Outer 1','Outer 2'],ylim=(0,115),ylabel='8-class window accuracy (%)',title='Same 200 ms input; record-shared development');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.2);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
