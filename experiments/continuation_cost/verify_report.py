"""Verify raw timing summaries, preserve source hashes, and render CPU costs."""
import hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
from benchmark import CASES
R=RUN_ROOT/'results/continuation_cost';P=RUN_ROOT/'private/continuation_cost'
if (R/'verification.json').exists() and '--render-only' not in sys.argv:raise FileExistsError('Preserve completed cost verification')
rows=[]
for case in CASES:
    item=json.loads((R/f'{case}.json').read_text());raw=json.loads((P/f'{case}_timings.json').read_text())
    assert item['device']=='cpu' and item['threads']==6
    for source in item['source_models']:
        path=RUN_ROOT/source['path'];assert path.stat().st_size==source['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    assert sum(a['bytes'] for a in item['source_models'])==item['model_bytes']
    for stage,times in raw.items():
        s=item['timing'][stage];assert len(times)==s['repeats']==200 and s['warmup']==20
        assert abs(np.median(times)-s['median_ms'])<1e-12
        assert abs(np.percentile(times,95)-s['p95_ms'])<1e-12
    rows.append(dict(case=case,training_trials=item['training_trials'],alpha=item['alpha'],
        model_mb=item['model_bytes']/1e6,fit_seconds=item['fit_seconds_sum'],
        total_median_ms=item['timing']['total_batch1']['median_ms'],total_p95_ms=item['timing']['total_batch1']['p95_ms'],
        feature_median_ms=item['timing']['feature_batch1']['median_ms'],
        scaler_model_median_ms=item['timing']['scaler_model_probability_batch1']['median_ms']))
table=pd.DataFrame(rows);table.to_csv(R/'summary.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for row,(stage,title) in enumerate([('inner','768 training trials / inner fold 0'),('outer','1,024 training trials / outer fold 0')]):
    part=table[table.case.str.startswith(stage+'_')];labels=part.case.str.replace(stage+'_','',regex=False)
    y=np.arange(len(part));times=part.total_median_ms.to_numpy();p95=part.total_p95_ms.to_numpy()
    axes[row,0].barh(y,times,color='#3377aa');axes[row,0].errorbar(times,y,xerr=[np.zeros(len(y)),p95-times],fmt='none',color='black',capsize=3)
    axes[row,1].barh(y,part.model_mb,color='#44aa88')
    for col,values in [(0,times),(1,part.model_mb.to_numpy())]:
        for yy,value,position in zip(y,values,p95 if col==0 else values):axes[row,col].text(position,yy,f' {value:.2f}',va='center',fontsize=9)
        axes[row,col].set(yticks=y,yticklabels=labels,title=title)
        axes[row,col].invert_yaxis();axes[row,col].grid(axis='x',alpha=.2)
        lo,hi=axes[row,col].get_xlim();axes[row,col].set_xlim(lo,hi*1.2)
    axes[row,0].set_xlabel('Feature + scaler + model, CPU ms (median / p95)')
    axes[row,1].set_xlabel('Actual model + scaler files, MB')
fig.savefig(R/'cpu_cost.png',dpi=170);plt.close(fig)
(R/'verification.json').write_text(json.dumps(dict(cases_verified=len(rows),raw_timing_summaries_recalculated=True,
    source_models_unchanged=True,new_model_fits=0,signal_acquisition_ms_excluded=200),indent=2)+'\n')
print(table.to_string(index=False))
