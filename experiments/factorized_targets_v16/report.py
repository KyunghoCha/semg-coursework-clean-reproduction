"""Aggregate classification and cost comparison, with no post-evaluation selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
from model_family import R


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    pooled=pd.read_csv(R/'pooled.csv');costs=json.loads((R/'costs.json').read_text());components=pd.read_csv(R/'components.csv')
    table=[]
    for row in costs:
        t=row['timing']['total_batch1'];table.append(dict(protocol=row['protocol'],method=row['method'],active_models=row['active_models'],trees=row['trees'],model_mb=row['model_bytes']/1e6,fit_seconds=row['fit_seconds_sum'],median_ms=t['median_ms'],p95_ms=t['p95_ms']))
    pd.DataFrame(table).to_csv(R/'cost_summary.csv',index=False)
    comparisons=[]
    for protocol in ['trial_shared','record_disjoint']:
        group=pooled[pooled.protocol==protocol].set_index('method');base=group.loc['direct8']
        for method in ['independent','conditional']:
            row=group.loc[method];comparisons.append(dict(protocol=protocol,method=method,correct_delta=int(row.correct-base.correct),accuracy_delta_pp=float((row.accuracy-base.accuracy)*100),angle_delta_pp=float((row.angle_accuracy-base.angle_accuracy)*100),gesture_delta_pp=float((row.gesture_accuracy-base.gesture_accuracy)*100),angle_only_error_delta=int(row.angle_only_errors-base.angle_only_errors),gesture_only_error_delta=int(row.gesture_only_errors-base.gesture_only_errors),both_wrong_delta=int(row.both_wrong-base.both_wrong)))
    report=dict(component_fits=48,family_evaluations=16,baseline_fits_reused=8,cumulative_observed_fit_records=406,cumulative_saved_models=186,component_fit_seconds=float(components.fit_seconds.sum()),cost_conditions=6,comparisons=comparisons,post_evaluation_model_selection=False,external_independent_test=False)
    (R/'summary.json').open('x').write(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    names=['direct8','independent','conditional'];labels=['Direct 8','Independent','Conditional']
    for ax,protocol in zip(axes,['trial_shared','record_disjoint']):
        group=pooled[pooled.protocol==protocol].set_index('method').loc[names]
        for j,(key,label,color) in enumerate([('accuracy','Final 8 classes','#4477aa'),('gesture_accuracy','Gesture from final class','#55aa88'),('angle_accuracy','Angle from final class','#dd9955')]):
            positions=np.arange(3)+(j-1)*.25;vals=group[key].to_numpy()*100
            ax.bar(positions,vals,width=.24,label=label,color=color)
            for pos,val in zip(positions,vals):ax.text(pos,val+1.1,f'{val:.1f}',ha='center',fontsize=7)
        ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,105),ylabel='Window accuracy (%)',title='Shared records' if protocol=='trial_shared' else 'Disjoint records');ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8,loc='upper left')
    fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    print(pooled[['protocol','method','accuracy','angle_accuracy','gesture_accuracy']].to_string(index=False));print(pd.DataFrame(table).to_string(index=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
