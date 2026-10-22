"""Verified aggregate gains and cost, without modifying the selected weights."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import P,R


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    pooled=pd.read_csv(R/'pooled.csv').set_index('role');outer=pd.read_csv(R/'outer.csv');paired=pd.read_csv(R/'paired_changes.csv');selection=json.loads((R/'selection.json').read_text());complete=json.loads((R/'complete.json').read_text());costs=json.loads((R/'costs.json').read_text());table=[]
    if costs:
        raw=json.loads((P/'cost_raw_timings.json').read_text())
        for row in costs:
            for label,timing in row['timing'].items():
                vals=raw[row['case']][label];assert len(vals)==200 and timing['median_ms']==float(np.median(vals)) and timing['p95_ms']==float(np.percentile(vals,95))
            t=row['timing']['total_batch1'];table.append(dict(case=row['case'],fold=row['fold'],alpha=row['alpha'],active_models=row['active_models'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds_sum'],median_ms=t['median_ms'],p95_ms=t['p95_ms']))
    pd.DataFrame(table,columns=['case','fold','alpha','active_models','model_mb','fit_seconds','median_ms','p95_ms']).to_csv(R/'cost_summary.csv',index=False)
    report=dict(new_fits=complete['new_fits'],inner_new_fits=0,fold_alphas=selection['fold_alphas'],final_development_alpha=selection['final_development_alpha'],baseline_accuracy=float(pooled.loc['baseline','accuracy']),selected_accuracy=float(pooled.loc['selected','accuracy']),accuracy_delta_pp=float(100*(pooled.loc['selected','accuracy']-pooled.loc['baseline','accuracy'])),correct_delta=int(pooled.loc['selected','correct']-pooled.loc['baseline','correct']),paired_changes={c:int(paired[c].sum()) for c in paired.columns if c!='fold'},cost_conditions=len(costs),raw_timing_verified=True,external_independent_test=False)
    (R/'summary.json').open('x').write(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained');x=np.arange(3)
    for offset,role,label,color in [(-.18,'baseline','XGB410','#4477aa'),(.18,'selected','Inner-selected XGB/LDA','#55aa88')]:
        rows=outer[outer.role==role].sort_values('fold');values=rows.accuracy.to_numpy()*100;ax.bar(x+offset,values,.35,label=label,color=color)
        for a,b in zip(x+offset,values):ax.text(a,b+1,f'{b:.3f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Outer 0','Outer 1','Outer 2'],ylim=(0,112),ylabel='8-class window accuracy (%)',title='Inner-selected small LDA mixture');ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2));print(pd.DataFrame(table).to_string(index=False))


if __name__=='__main__':main()
