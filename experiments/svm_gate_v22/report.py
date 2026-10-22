"""Report identical-label checks and the full latency distribution summaries."""
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT,SOURCE_ROOT
P=RUN_ROOT/'private/svm_gate_v22';R=RUN_ROOT/'results/svm_gate_v22'


def main():
    assert (R/'verification.json').exists() and (R/'runtime_verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve gate report')
    decisions=pd.read_csv(R/'decisions.csv');costs=json.loads((R/'costs.json').read_text());raw=json.loads((P/'cost_raw_timings.json').read_text());ct=[]
    for row in costs:
        for label,t in row['timing'].items():
            v=raw[row['method']][label];assert len(v)==200 and t['median_ms']==float(np.median(v)) and t['mean_ms']==float(np.mean(v)) and t['p95_ms']==float(np.percentile(v,95))
        t=row['timing']['total_batch1'];ct.append(dict(method=row['method'],fold=row['fold'],model_mb=row['model_bytes']/1e6,median_ms=t['median_ms'],mean_ms=t['mean_ms'],p95_ms=t['p95_ms'],measured_calls=t['svm_calls'],measured_windows=200))
    pd.DataFrame(ct).to_csv(R/'cost_summary.csv',index=False);agg=decisions.groupby('stage')[['windows','svm_calls','skipped','correct','label_differences']].sum();agg['call_fraction']=agg.svm_calls/agg.windows;agg['accuracy']=agg.correct/agg.windows;agg.to_csv(R/'stage_summary.csv');summary=dict(new_model_fits=0,threshold_searches=0,stages=agg.reset_index().to_dict('records'),runtime_outer_cpu_windows=29184,all_labels_identical=True,full_probabilities_preserved=False,models_unchanged=True,model_size_reduced=False,cost_conditions=len(costs),raw_cost_times_recomputed=True,external_independent_test=False)
    ev=json.loads((R/'evaluation_cost.json').read_text());raw_path=P/'evaluation_cost_raw.npz';assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==ev['raw_timing_sha256'];evraw=np.load(raw_path);v=evraw['milliseconds'];mask=evraw['svm_called'];assert len(v)==9728 and int(mask.sum())==792
    plan=SOURCE_ROOT/'experiments/svm_gate_v22/EVALUATION_COST_PLAN.md';assert hashlib.sha256(plan.read_bytes()).hexdigest()==json.loads((R/'evaluation_cost_started.json').read_text())['plan_sha256']
    for key,a in [('all',v),('svm_called',v[mask]),('svm_skipped',v[~mask])]:
        expected=dict(count=len(a),median_ms=float(np.median(a)),mean_ms=float(np.mean(a)),p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),max_ms=float(np.max(a)))
        assert ev[key]==expected
    summary['all_outer_fold0_timing']=ev['all'];summary['all_outer_fold0_call_fraction']=ev['call_fraction'];summary['benchmark_call_fraction']=ct[1]['measured_calls']/200
    (R/'evaluation_timing_verification.json').open('x').write(json.dumps(dict(plan_hash_unchanged=True,raw_hash_matches=True,all_9728_times_recomputed=True,conditional_counts_match=True,benchmark_and_outer_scopes_distinguished=True),indent=2)+'\n')
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.5,3.4),layout='constrained');x=np.arange(2)
    for offset,key,label in [(-.25,'median_ms','Median'),(0,'mean_ms','Mean'),(.25,'p95_ms','p95')]:
        vals=[r[key] for r in ct];ax.bar(x+offset,vals,.23,label=label)
        for a,b in zip(x+offset,vals):ax.text(a,b+.5,f'{b:.2f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Full mixture labels','Safe SVM gate labels'],ylabel='Feature + classification latency (ms)',title='Same 200 benchmark windows: 2.5% gated SVM calls');ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);fig.savefig(R/'latency.png',dpi=170);plt.close(fig)
    bins=np.arange(0,np.ceil(v.max())+2,1);nc,_=np.histogram(v[mask],bins);ns,_=np.histogram(v[~mask],bins);pd.DataFrame(dict(left_ms=bins[:-1],right_ms=bins[1:],svm_called=nc,svm_skipped=ns)).to_csv(R/'evaluation_latency_histogram.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');ax.stairs(ns,bins,label='SVM skipped: 8,936 windows',fill=True,alpha=.55);ax.stairs(nc,bins,label='SVM called: 792 windows',fill=True,alpha=.65);ax.set(xlabel='Feature + label latency (ms)',ylabel='Window count (log scale)',yscale='log',ylim=(.8,15000),title='All 9,728 outer-fold0 windows: 8.14% SVM calls');ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2);fig.savefig(R/'evaluation_latency.png',dpi=170);plt.close(fig);print(json.dumps(summary,indent=2));print(pd.DataFrame(ct).to_string(index=False))


if __name__=='__main__':main()
