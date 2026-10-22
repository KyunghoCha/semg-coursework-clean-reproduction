"""Aggregate failed or successful predeclared comparison without selecting new settings."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lda_common import P,R


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    scores=pd.read_csv(R/'scores.csv');inner=pd.read_csv(R/'inner_overlap.csv');selection=json.loads((R/'selection.json').read_text());costs=json.loads((R/'costs.json').read_text());raw=json.loads((P/'cost_raw_timings.json').read_text());ct=[]
    for row in costs:
        for label,timing in row['timing'].items():
            values=np.array(raw[row['case']][label]);assert len(values)==200
            assert timing['median_ms']==float(np.median(values)) and timing['p95_ms']==float(np.percentile(values,95))
        t=row['timing']['total_batch1'];ct.append(dict(case=row['case'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds'],fit_device=row['fit_device'],median_ms=t['median_ms'],p95_ms=t['p95_ms']))
    pd.DataFrame(ct).to_csv(R/'cost_summary.csv',index=False)
    report=dict(new_fits=len(scores),inner_fits=3,outer_fits=int((scores.stage=='outer').sum()),inner_evaluation_observations=int(inner.windows.sum()),inner_is_complete_oof=False,baseline_inner_accuracy=float(inner.baseline_correct.sum()/inner.windows.sum()),lda_inner_accuracy=float(inner.lda_correct.sum()/inner.windows.sum()),inner_fold_choices=selection['choices'],cost_conditions=len(costs),raw_cost_timings_recomputed=True,external_independent_test=False)
    for key in ['both_correct','lda_only_correct','baseline_only_correct','both_wrong']:report[key]=int(inner[key].sum())
    (R/'summary.json').open('x').write(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.5,3.6),layout='constrained');x=np.arange(3)
    for off,key,label,col in [(-.18,'baseline_correct','XGB410, 64 targets','#4477aa'),(.18,'lda_correct','Ledoit-Wolf LDA410, 64 targets','#cc7755')]:
        values=100*inner[key]/inner.windows;ax.bar(x+off,values,.35,label=label,color=col)
        for a,b in zip(x+off,values):ax.text(a,b+1,f'{b:.2f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Inner 0','Inner 1','Inner 2'],ylabel='8-class window accuracy (%)',ylim=(0,115),title='Shared covariance boundary comparison');ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2));print(pd.DataFrame(ct).to_string(index=False))


if __name__=='__main__':main()
