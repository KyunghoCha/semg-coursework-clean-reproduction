"""Verify raw timing summaries and report the separate, longer-input condition."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context_inputs import P,R,assert_preserved


def stats(a):
    return dict(count=len(a),median_ms=float(np.median(a)),mean_ms=float(np.mean(a)),p95_ms=float(np.percentile(a,95)))


def main():
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    verification=json.loads((R/'verification.json').read_text());cost=json.loads((R/'costs.json').read_text());assert_preserved(cost['source_hashes']);raw=np.load(P/'cost_raw.npz');ct=[]
    import hashlib
    assert hashlib.sha256((P/'cost_raw.npz').read_bytes()).hexdigest()==cost['raw_timing_sha256']
    for row in cost['conditions']:
        k=row['history']
        for name,prefix in [('full_path','full'),('averaging_only','mean')]:assert row[name]==stats(raw[f'{prefix}_k{k}'])
        ct.append(dict(history=k,maximum_signal_span_ms=row['maximum_signal_span_ms'],model_mb=cost['source_model_bytes']/1e6,original_training_seconds_sum=cost['original_training_seconds_sum'],**row['full_path'],averaging_only_median_ms=row['averaging_only']['median_ms'],averaging_only_p95_ms=row['averaging_only']['p95_ms'],history_payload_bytes=row['state']['array_payload_bytes'],state_bytes_excluding_shared_keys_and_integers=row['state']['persistent_total_excluding_shared_keys_and_integers_bytes']))
    pd.DataFrame(ct).to_csv(R/'cost_summary.csv',index=False)
    pooled=pd.read_csv(R/'pooled.csv');paired=pd.read_csv(R/'paired_changes.csv');positions=pd.read_csv(R/'positions.csv').groupby(['role','position'],as_index=False).agg(windows=('windows','sum'),correct=('correct','sum'),actual_signal_span_ms=('actual_signal_span_ms','first'),start_ms=('start_ms','first'));positions['accuracy']=positions.correct/positions.windows;positions.to_csv(R/'position_summary.csv',index=False)
    runs=pd.read_csv(R/'error_runs.csv').groupby(['role','run_length'],as_index=False).runs.sum();runs['wrong_windows']=runs.runs*runs.run_length;runs.to_csv(R/'error_run_summary.csv',index=False);axes=pd.read_csv(R/'error_axes.csv').groupby('role',as_index=False)[['angle_only','gesture_only','both']].sum();axes.to_csv(R/'error_axis_summary.csv',index=False)
    for role in ['current','selected']:
        row=pooled[pooled.role==role].iloc[0];assert int(runs[runs.role==role].wrong_windows.sum())==int(row.windows-row.correct)
        assert int(axes[axes.role==role][['angle_only','gesture_only','both']].sum(axis=1).iloc[0])==int(row.windows-row.correct)
    selection=json.loads((R/'selection.json').read_text());summary=dict(outer=pooled.to_dict('records'),histories=selection['fold_histories'],maximum_signal_span_ms=600,first_window_label_changes=0,original_200ms_primary_accuracy=float(pooled[pooled.role=='current'].accuracy.iloc[0]),new_model_fits=0,all_1536_development_trials_and_19_windows_retained=True,known_boundary_reset_required=True,future_wait_ms=0,transition_response_delay_measured=False,v22_gate_used=False,external_independent_test=False,source_record_dependence_unresolved=True,raw_timing_statistics_verified=True,cost=ct)
    for c in ['corrected','broken','net','trial_improved','trial_worsened','trial_tied']:summary[c]=int(paired[c].sum())
    summary['axis_errors']=axes.to_dict('records');summary['single_window_error_runs']={role:int(runs[(runs.role==role)&(runs.run_length==1)].runs.iloc[0]) for role in ['current','selected']};summary['verification']=verification
    (R/'timing_verification.json').open('x').write(json.dumps(dict(raw_sha256_preserved=True,conditions=len(ct),statistics_recomputed=True,all_full_path_measurements=380*len(ct),all_averaging_measurements=9728*len(ct),model_hashes_preserved=True),indent=2)+'\n')
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    labels={'current':'Current only: 200 ms','selected':'Past + current: up to 600 ms'};colors={'current':'#4477aa','selected':'#cc6677'}
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained');out=pd.read_csv(R/'outer.csv');x=np.arange(3)
    for off,role in [(-.18,'current'),(.18,'selected')]:
        vals=100*out[out.role==role].sort_values('fold').accuracy.to_numpy();ax.bar(x+off,vals,.35,label=labels[role],color=colors[role])
        for a,b in zip(x+off,vals):ax.text(a,b+.5,f'{b:.2f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['Outer fold 0','Outer fold 1','Outer fold 2'],ylim=(0,118),ylabel='8-class window accuracy (%)',title='Separate input-time conditions; record-shared development');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.2);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,6.5),layout='constrained',sharex=True)
    for role in ['current','selected']:
        p=positions[positions.role==role];axs[0].plot(p.position+1,p.accuracy*100,'o-',label=labels[role],color=colors[role]);axs[1].step(p.position+1,p.actual_signal_span_ms,where='mid',label=labels[role],color=colors[role])
    axs[0].set(ylabel='Window accuracy (%)',ylim=(70,102),title='All 19 windows included; first window unchanged; reset each trial');axs[0].legend(fontsize=9);axs[1].set(xlabel='Window position (100 ms stride)',ylabel='Input time span (ms)',xticks=np.arange(1,20),yticks=[200,300,400,500,600]);[a.grid(alpha=.2) for a in axs];fig.savefig(R/'positions.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for off,role in [(-.19,'current'),(.19,'selected')]:
        part=runs[runs.role==role];ax.bar(part.run_length+off,part.runs,.37,label=labels[role],color=colors[role])
    ax.set(xlabel='Consecutive incorrect outputs within one trial',ylabel='Number of error runs',xticks=range(1,20),title='Error-run distribution; overlapping windows are correlated');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.savefig(R/'error_runs.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
