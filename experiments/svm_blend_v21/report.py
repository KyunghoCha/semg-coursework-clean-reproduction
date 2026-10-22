"""Aggregate only verified v21 probabilities and measured cost; no retuning."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private/svm_blend_v21';R=RUN_ROOT/'results/svm_blend_v21'


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    pooled=pd.read_csv(R/'pooled.csv');paired=pd.read_csv(R/'paired_changes.csv');complete=json.loads((R/'complete.json').read_text());sel=json.loads((R/'selection.json').read_text());costs=json.loads((R/'costs.json').read_text());ct=[]
    if costs:
        raw=json.loads((P/'cost_raw_timings.json').read_text())
        for row in costs:
            for key,t in row['timing'].items():
                vals=raw[row['method']][key];assert len(vals)==200 and t['median_ms']==float(np.median(vals)) and t['p95_ms']==float(np.percentile(vals,95))
            t=row['timing']['total_batch1'];ct.append(dict(method=row['method'],fold=row['fold'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds'],median_ms=t['median_ms'],p95_ms=t['p95_ms']))
    pd.DataFrame(ct).to_csv(R/'cost_summary.csv',index=False);summary=dict(**complete,fold_betas=sel['fold_betas'],outer=pooled.to_dict('records'),cost_conditions=len(costs),raw_cost_times_recomputed=True,external_independent_test=False)
    for c in ['corrected','broken','net','trial_improved','trial_worsened','trial_tied']:summary[c]=int(paired[c].sum())
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n');fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained');out=pd.read_csv(R/'outer.csv');x=np.arange(3)
    for off,role,label,col in [(-.18,'incumbent','Fixed XGB/LDA','#4477aa'),(.18,'selected','Inner-selected SVM mixture','#aa7799')]:
        vals=100*out[out.role==role].sort_values('fold').accuracy.to_numpy();ax.bar(x+off,vals,.35,label=label,color=col)
        for a,b in zip(x+off,vals):ax.text(a,b+1,f'{b:.2f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Outer 0','Outer 1','Outer 2'],ylim=(0,115),ylabel='8-class window accuracy (%)',title='Frozen small SVM mixture: development only');ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig);print(json.dumps(summary,indent=2));print(pd.DataFrame(ct).to_string(index=False))


if __name__=='__main__':main()
