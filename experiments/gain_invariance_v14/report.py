"""Aggregate matched gain-control evidence without new predictions or selection."""
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/gain_invariance_v14';P=RUN_ROOT/'private/gain_invariance_v14'
if (R/'matched_summary.csv').exists():raise FileExistsError('Preserve completed report')
rows=[]
for kind,stage in [('raw25','active_record_baseline'),('normalized25','gain_invariance_v14')]:
    assert (RUN_ROOT/f'results/{stage}/verification.json').exists()
    assert json.loads((RUN_ROOT/f'private/{stage}/splits.json').read_text())==json.loads((P/'splits.json').read_text())
    table=pd.read_csv(RUN_ROOT/f'results/{stage}/pooled.csv');table.insert(0,'representation',kind);rows.append(table)
summary=pd.concat(rows,ignore_index=True);summary.to_csv(R/'matched_summary.csv',index=False)
paired=[]
for protocol in ['trial_shared','record_disjoint']:
    raw_only=gain_only=both=neither=0
    for fold in range(4):
        a=np.load(RUN_ROOT/f'private/active_record_baseline/{protocol}_{fold}.npz');b=np.load(P/f'{protocol}_{fold}.npz')
        for k in ['truth','trial']:np.testing.assert_array_equal(a[k],b[k])
        ca=a['truth']==a['prediction'];cb=b['truth']==b['prediction'];raw_only+=int((ca&~cb).sum());gain_only+=int((~ca&cb).sum());both+=int((ca&cb).sum());neither+=int((~ca&~cb).sum())
    paired.append(dict(protocol=protocol,raw_only=raw_only,normalized_only=gain_only,both=both,neither=neither,windows=raw_only+gain_only+both+neither))
pd.DataFrame(paired).to_csv(R/'paired_windows.csv',index=False)
cost=[]
for kind in ['raw25','normalized25']:
    for protocol in ['trial_shared','record_disjoint']:
        case=kind+'_'+protocol;report=json.loads((R/f'cost_{case}.json').read_text());raw=json.loads((P/f'cost_{case}_timings.json').read_text())
        for key,t in report['timing'].items():
            assert len(raw[key])==200
            np.testing.assert_allclose([t['median_ms'],t['p95_ms']],[np.median(raw[key]),np.percentile(raw[key],95)],rtol=0,atol=0)
        stage='active_record_baseline' if kind=='raw25' else 'gain_invariance_v14'
        path=RUN_ROOT/f'private/{stage}/{protocol}_0.joblib';assert hashlib.sha256(path.read_bytes()).hexdigest()==report['model_sha256']
        cost.append(dict(representation=kind,protocol=protocol,train_trials=report['training_trials'],model_mb=report['model_bytes']/1e6,fit_seconds=report['fit_seconds'],feature_median_ms=report['timing']['feature_batch1']['median_ms'],total_median_ms=report['timing']['total_batch1']['median_ms'],total_p95_ms=report['timing']['total_batch1']['p95_ms']))
pd.DataFrame(cost).to_csv(R/'cost_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(8.2,4.5),layout='constrained')
for j,(kind,label,color) in enumerate([('raw25','Original 25 features','#3377aa'),('normalized25','Common window-RMS normalization','#dd7744')]):
    for i,protocol in enumerate(['trial_shared','record_disjoint']):
        row=summary[(summary.representation==kind)&(summary.protocol==protocol)].iloc[0];x=i+(j-.5)*.32;y=row.accuracy*100
        ax.bar(x,y,width=.29,color=color,label=label if i==0 else None);ax.errorbar(x,y,yerr=row.fold_std*100,fmt='none',color='black',capsize=3);ax.text(x,y+row.fold_std*100+1.5,f'{y:.2f}%',ha='center')
ax.axhline(12.5,ls='--',color='gray',lw=1,label='Uniform random expectation: 12.5%')
ax.set(xticks=[0,1],xticklabels=['Disjoint trials, shared recordings','Disjoint recordings'],ylim=(0,79),ylabel='All activity-window accuracy (%)',title='Same direct-8 XGB400 / same splits / all 19 windows')
ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.2);fig.savefig(R/'gain_transfer.png',dpi=170);plt.close(fig)
(R/'report_verification.json').write_text(json.dumps(dict(splits_identical=True,windows_paired_per_protocol=29184,source_models_preserved=True,cost_cases_verified=4,raw_timing_recalculated=True,new_fits_in_report=0,posthoc_diagnostic=True,external_independent_data=False),indent=2)+'\n')
print(summary.to_string(index=False));print(pd.DataFrame(cost).to_string(index=False))
