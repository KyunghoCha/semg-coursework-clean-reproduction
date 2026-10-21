"""Recalculate timing summaries and compare Pareto points only within shared evaluations."""
from semg.paths import RUN_ROOT
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import SPECS
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'
assert (R/'benchmark_complete.json').exists()
base=pd.read_csv(RUN_ROOT/'results/baseline.csv');base=base[base.phase=='baseline_holdout']
v5=pd.read_csv(RUN_ROOT/'results/latent_subclass_v5/pooled.csv')
v4=pd.read_csv(RUN_ROOT/'results/capacity_v4/outer_pooled.csv')
v4i=pd.read_csv(RUN_ROOT/'results/capacity_v4/inner.csv')
v6=pd.read_csv(RUN_ROOT/'results/cross_spectrum_v6/pooled.csv')
v7=pd.read_csv(RUN_ROOT/'results/cross_capacity_v7/pooled.csv')
groups={'lecture_holdout':base.groupby('model').accuracy.mean().to_dict(),'latent_v5':v5.set_index('model').accuracy.to_dict(),
    'capacity_v4':v4.set_index('role').accuracy.to_dict(),'capacity_v4_inner':v4i.groupby('candidate').accuracy.mean().to_dict(),
    'cross_v6':v6.set_index('model').accuracy.to_dict(),'cross_v7':v7.set_index('model').accuracy.to_dict()}
keys=dict(lecture_SVM='SVM',lecture_RF100='RF',lecture_XGB400='XGB',rich8_400_v5='global8',rich64_400_v5='subclass64',
    rich8_800_v4='reference',rich8_1600_v4='selected',rich8_3200_v6='global8_3200',rich64_400_v6='subclass64_400',
    cross64_400_v6='subclass64_cross400',cross64_400_v7='cross400',cross64_800_v7='cross800',without_high64_400_v7='without_high_cross400',
    inner_rich800='XGB_rich800',inner_lean800='XGB_noTime800',inner_SVM_standard='SVM_rich_standard',inner_SVM_quantile='SVM_rich_quantile')
if (RUN_ROOT/'results/fine_spectrum_v8/verification.json').exists():
    fine=pd.read_csv(RUN_ROOT/'results/fine_spectrum_v8/pooled.csv')
    groups['cross_v7']['fine400']=float(fine.loc[fine.model=='fine400','accuracy'].iloc[0])
    keys['fine64_400_v8']='fine400'
assert all(s['selected']=='XGB_rich1600' for s in json.loads((RUN_ROOT/'results/capacity_v4/selections.json').read_text()))
rows=[];verified=0
for spec in SPECS:
    devices=['cpu'] if 'SVM' in spec['name'] or spec['name']=='lecture_RF100' else ['cuda','cpu']
    for device in devices:
        name=f"{spec['name']}_{device}";z=json.loads((R/f'{name}.json').read_text());raw=json.loads((P/f'{name}_timings.json').read_text())
        assert z['other_compute_processes_before']==z['other_compute_processes_after']==0
        for stage,values in raw.items():
            a=np.array(values);saved=z['timing'][stage];n=30 if stage.endswith('batch256') else 200
            assert len(a)==n and np.isfinite(a).all() and (a>0).all()
            assert abs(np.median(a)-saved['median_ms'])<1e-10 and abs(np.percentile(a,95)-saved['p95_ms'])<1e-10
            assert abs(sum(values)-saved['measured_total_ms'])<1e-8
            verified+=1
        row=dict(model=spec['name'],comparison_group=spec['group'],device=device,
            accuracy=groups[spec['group']][keys[spec['name']]],model_mb=z['model_bytes_with_scaler']/1e6,features=z['features'],
            feature_ms=z['timing']['feature_batch1']['median_ms'],scaler_ms=z['timing']['scaler_batch1']['median_ms'],
            model_median_ms=z['timing']['model_batch1']['median_ms'],model_p95_ms=z['timing']['model_batch1']['p95_ms'],
            total_median_ms=z['timing']['total_batch1']['median_ms'],total_p95_ms=z['timing']['total_batch1']['p95_ms'],
            context_update_median_ms=z['timing']['context_update_batch1']['median_ms'],
            total_context_median_ms=z['timing']['total_with_context_batch1']['median_ms'],
            batch1_windows_per_second=z['timing']['total_batch1']['windows_per_second'],
            model_batch256_windows_per_second=z['timing']['model_batch256']['windows_per_second'],
            total_batch256_windows_per_second=z['timing']['total_batch256']['windows_per_second'],**z['memory'])
        rows.append(row)
    if len(devices)==2:
        np.testing.assert_array_equal(np.load(P/f"{spec['name']}_cpu_prediction.npy"),np.load(P/f"{spec['name']}_cuda_prediction.npy"))
frame=pd.DataFrame(rows);frame['empirical_pareto']=True
for (_,device),g in frame.groupby(['comparison_group','device']):
    for idx,row in g.iterrows():
        weak=(g.accuracy>=row.accuracy)&(g.model_mb<=row.model_mb)&(g.total_median_ms<=row.total_median_ms)
        strict=(g.accuracy>row.accuracy)|(g.model_mb<row.model_mb)|(g.total_median_ms<row.total_median_ms)
        frame.loc[idx,'empirical_pareto']=not bool((weak&strict).any())
frame.to_csv(R/'accuracy_cost.csv',index=False)
(R/'verification.json').write_text(json.dumps(dict(model_device_pairs=len(frame),timing_series_recomputed=verified,
    gpu_sync_verified=True,cpu_gpu_prediction_equivalence=True,prediction_equivalence_windows=256,
    prediction_equivalence_scope='fixed benchmark inputs, not all evaluation windows',comparison_groups_separate=True,
    memory_scope='process RSS and two observed VRAM snapshots; not training peak',
    pareto_scope='descriptive measured medians, not a statistical dominance test'),indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');g=frame[frame.comparison_group=='cross_v7']
labels={'cross64_400_v7':'Cross / 400 rounds','cross64_800_v7':'Cross / 800 rounds','without_high64_400_v7':'Without high-band relations','fine64_400_v8':'Cross + fine power / 400'}
for i,(name,a) in enumerate(g.groupby('model')):
    color=f'C{i}'
    for row in a.itertuples():
        axes[0].scatter(row.total_median_ms,row.accuracy*100,color=color,marker='^' if row.device=='cuda' else 'o',s=75,
                        label=f'{labels[name]} / {row.device}')
    row=a.iloc[0];axes[1].scatter(row.model_mb,row.accuracy*100,color=color,s=90)
    axes[1].annotate(labels[name],(row.model_mb,row.accuracy*100),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
axes[0].set(xlabel='Feature + scaler + model latency, batch1 median (ms)',ylabel='Development window accuracy (%)',title='Same fixed evaluation; CPU/GPU separated')
axes[0].set_xscale('log')
axes[0].legend(fontsize=7)
axes[1].set(xlabel='Saved model + scaler (MB)',ylabel='Development window accuracy (%)',title='Measured size and accuracy')
for ax in axes:ax.margins(.25);ax.grid(alpha=.2)
fig.savefig(R/'accuracy_latency_size.png',dpi=170);plt.close(fig)
print(frame[['model','device','accuracy','model_mb','total_median_ms','total_p95_ms','empirical_pareto']].to_string(index=False))
