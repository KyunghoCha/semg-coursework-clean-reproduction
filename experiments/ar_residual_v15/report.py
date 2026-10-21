"""Aggregate-only AR comparison and one figure; no new model selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/ar_residual_v15'


def main():
    verify=json.loads((R/'verification.json').read_text());cost=json.loads((R/'costs.json').read_text())
    if (R/'summary.json').exists():raise FileExistsError('Preserve completed report')
    rows=pd.read_csv(R/'scores.csv');outer=pd.read_csv(R/'outer_comparison.csv');control=pd.read_csv(R/'control_pooled.csv')
    base=pd.read_csv(RUN_ROOT/'results/active_record_baseline/pooled.csv').assign(model='td25')
    table=pd.concat([base,control],ignore_index=True)
    table[['model','protocol','correct','windows','accuracy','fold_std','mean_model_mb','mean_fit_seconds']].to_csv(R/'comparison.csv',index=False)
    inner=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/inner.csv');inner=inner[inner.model=='narrow64'][['model','fold','correct','evaluation_windows','accuracy']]
    inner=pd.concat([inner,rows[rows.stage=='inner'][['model','fold','correct','evaluation_windows','accuracy']]],ignore_index=True)
    inner.to_csv(R/'inner_comparison.csv',index=False)
    inners=[]
    for name,part in inner.groupby('model'):
        inners.append(dict(model=name,correct=int(part.correct.sum()),windows=int(part.evaluation_windows.sum()),accuracy=float(part.correct.sum()/part.evaluation_windows.sum()),fold_sd=float(part.accuracy.std(ddof=0))))
    summary=dict(new_fits=len(rows),total_observed_fit_records_through_v15=335+len(rows),total_saved_models_through_v15=115+len(rows),
                 inner=inners,outer_correct=int(outer.correct.sum()),outer_baseline_correct=int(outer.baseline_correct.sum()),outer_windows=int(outer.windows.sum()),
                 outer_accuracy=float(outer.correct.sum()/outer.windows.sum()),outer_baseline_accuracy=float(outer.baseline_correct.sum()/outer.windows.sum()),
                 outer_fold_sd=float(outer.accuracy.std(ddof=0)),outer_choices=outer.selected.tolist(),
                 new_fit_seconds_sum=float(rows.fit_seconds.sum()),cost_cases=len(cost),
                 trial_improved=verify['trial_improved'],trial_worse=verify['trial_worse'],trial_tied=verify['trial_tied'],
                 independent_external_test=False)
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    costs=[]
    for row in cost:
        costs.append(dict(case=row['case'],features=row['features'],classes=row['classes'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds'],median_ms=row['timing']['total_batch1']['median_ms'],p95_ms=row['timing']['total_batch1']['p95_ms']))
    pd.DataFrame(costs).to_csv(R/'cost_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    names=['td25','td_ar','td_ar_res'];colors=['#4477aa','#ee8844']
    for j,protocol in enumerate(['trial_shared','record_disjoint']):
        part=table[table.protocol==protocol].set_index('model').loc[names]
        positions=np.arange(3)+(j-.5)*.36
        axes[0].bar(positions,part.accuracy*100,width=.34,color=colors[j],label=['Shared records','Disjoint records'][j])
        for x,v in zip(positions,part.accuracy*100):axes[0].text(x,v+1.2,f'{v:.2f}',ha='center',fontsize=8)
    axes[0].set(xticks=np.arange(3),xticklabels=['TD25','TD25 + AR6','TD25 + AR6\n+ residual RMS'],ylim=(0,104),ylabel='Window accuracy (%)',title='Direct 8-class XGB400 / same 4 folds')
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    values=[summary['outer_baseline_accuracy']*100,summary['outer_accuracy']*100]
    axes[1].bar([0,1],values,color=['#4477aa','#55aa88'],width=.6)
    for i,v in enumerate(values):axes[1].text(i,v+1.2,f'{v:.4f}%',ha='center')
    axes[1].set(xticks=[0,1],xticklabels=['Fixed narrow410','Inner-selected AR option'],ylim=(0,104),ylabel='Window accuracy (%)',title='Record-associated 64 targets / 3 outer folds')
    axes[1].grid(axis='y',alpha=.2)
    fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));print(table.to_string(index=False));print(pd.DataFrame(costs).to_string(index=False))


if __name__=='__main__':main()
