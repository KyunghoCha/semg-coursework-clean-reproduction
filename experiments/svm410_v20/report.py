"""SVM comparison and internal error diagnostics, retaining all windows and scope."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared20 import P,R


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    scores=pd.read_csv(R/'scores.csv');inner=pd.read_csv(R/'inner_overlap.csv');complete=json.loads((R/'complete.json').read_text());selection=json.loads((R/'selection.json').read_text());costs=json.loads((R/'costs.json').read_text());raw=json.loads((P/'cost_raw_timings.json').read_text());ct=[]
    for row in costs:
        for label,t in row['timing'].items():
            vals=raw[row['case']][label];assert len(vals)==200 and t['median_ms']==float(np.median(vals)) and t['p95_ms']==float(np.percentile(vals,95))
        t=row['timing']['total_batch1'];ct.append(dict(case=row['case'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds'],median_ms=t['median_ms'],p95_ms=t['p95_ms']))
    pd.DataFrame(ct).to_csv(R/'cost_summary.csv',index=False)
    summary=dict(**complete,incumbent_inner_accuracy=float(inner.incumbent_correct.sum()/inner.windows.sum()),svm_inner_accuracy=float(inner.svm_correct.sum()/inner.windows.sum()),inner_observations=int(inner.windows.sum()),inner_is_full_oof=False,choices=selection['choices'],calibrated_fit_seconds=float(scores.calibrated_model_fit_seconds.sum()),base_svc_fit_seconds=float(scores.base_svc_fit_seconds.sum()),cost_conditions=len(costs),raw_cost_times_recomputed=True,existing_240_feature_svm_is_not_410_feature_evidence=True,external_independent_test=False)
    for c in ['both_correct','svm_only_correct','incumbent_only_correct','both_wrong']:summary[c]=int(inner[c].sum())
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained');x=np.arange(3)
    for off,key,label,col in [(-.18,'incumbent_correct','Fixed XGB/LDA410','#4477aa'),(.18,'svm_correct','Trial-calibrated SVM410','#aa7799')]:
        v=100*inner[key]/inner.windows;ax.bar(x+off,v,.35,label=label,color=col)
        for a,b in zip(x+off,v):ax.text(a,b+1,f'{b:.2f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Inner 0','Inner 1','Inner 2'],ylim=(0,115),ylabel='8-class window accuracy (%)',title='One fixed RBF SVM on 410 features');ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    pos=pd.read_csv(R/'internal_positions.csv').groupby('start_ms')[['windows','correct']].sum();energy=pd.read_csv(R/'internal_energy.csv').groupby('train_rms_quartile')[['windows','correct']].sum()
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained');axes[0].plot(pos.index/1000,100*pos.correct/pos.windows,'o-');axes[0].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',ylim=(0,100),title='Frozen v19: inner observations');axes[1].bar(np.arange(4)+1,100*energy.correct/energy.windows,color='#55aa88');axes[1].set(xlabel='RMS quartile from inner training',ylabel='Window accuracy (%)',ylim=(0,100),xticks=[1,2,3,4]);fig.savefig(R/'internal_diagnostic.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));print(pd.DataFrame(ct).to_string(index=False))


if __name__=='__main__':main()
