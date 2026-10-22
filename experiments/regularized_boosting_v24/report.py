"""Verified aggregate accuracy and matched cost; no outer-driven changes."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import P,R,digest,check_hashes


def main():
    assert (R/'verification.json').exists()
    if (R/'summary.json').exists():raise FileExistsError('Preserve report')
    costs=json.loads((R/'costs.json').read_text());raw=json.loads((P/'cost_raw.json').read_text());assert digest(P/'cost_raw.json')==costs['raw_timing_sha256'];check_hashes(costs['source_hashes'])
    for row in costs['rows']:
        v=raw[row['candidate']+'_'+row['mode']];assert len(v)==380
        for key,value in dict(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))).items():assert value==row[key]
    fits=pd.read_csv(R/'fits.csv');inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');pooled=pd.read_csv(R/'pooled.csv');paired=pd.read_csv(R/'paired_changes.csv');choices=json.loads((R/'selection.json').read_text())['fold_candidates'];totals=inner.groupby('candidate',as_index=False)[['correct','windows']].sum();totals['accuracy']=totals.correct/totals.windows;totals.to_csv(R/'inner_summary.csv',index=False)
    positions=pd.read_csv(R/'positions.csv').groupby(['role','position'],as_index=False)[['windows','correct']].sum();positions['accuracy']=positions.correct/positions.windows;positions.to_csv(R/'position_summary.csv',index=False)
    summary=dict(**json.loads((R/'complete.json').read_text()),fold_candidates=choices,inner=totals.to_dict('records'),outer=pooled.to_dict('records'),fit_seconds_sum=float(fits.fit_seconds.sum()),scaler_fit_seconds_sum=float(fits.scaler_fit_seconds.sum()),model_bytes_sum=int(fits.xgb_model_bytes.sum()),new_model_cost_measurements=costs['rows'],input_ms=200,temporal_context=False,v22_gate_used=False,all_19_windows_evaluated=True,external_independent_test=False,raw_cost_statistics_recomputed=True)
    for key in ['corrected','broken','net','trial_improved','trial_worsened','trial_tied']:summary[key]=int(paired[key].sum())
    (R/'summary.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');colors={'incumbent':'#4477aa','reg400':'#cc6677','shallow800':'#228833'}
    for offset,name in zip([-.25,0,.25],['incumbent','reg400','shallow800']):
        vals=inner[inner.candidate==name].sort_values('fold').accuracy.to_numpy()*100;axs[0].bar(np.arange(3)+offset,vals,.24,label=name,color=colors[name])
        for a,b in zip(np.arange(3)+offset,vals):axs[0].text(a,b+.5,f'{b:.2f}',ha='center',fontsize=7)
    for offset,role,label in [(-.18,'incumbent','Fixed v21'),(.18,'selected','Inner-selected XGB replacement')]:
        vals=outer[outer.role==role].sort_values('fold').accuracy.to_numpy()*100;axs[1].bar(np.arange(3)+offset,vals,.35,label=label)
        for a,b in zip(np.arange(3)+offset,vals):axs[1].text(a,b+.5,f'{b:.2f}',ha='center',fontsize=8)
    for ax,title in zip(axs,['Inner comparison: same 200 ms','Frozen outer assessment: same 200 ms']):ax.set(xticks=np.arange(3),xticklabels=['Fold 0','Fold 1','Fold 2'],ylim=(0,115),ylabel='8-class window accuracy (%)',title=title);ax.legend(loc='upper left',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for role,label in [('incumbent','Fixed v21'),('selected','Inner-selected XGB replacement')]:
        part=positions[positions.role==role];ax.plot(part.position+1,100*part.accuracy,'o-',label=label)
    ax.set(xlabel='Window position (100 ms stride; each input 200 ms)',ylabel='8-class accuracy (%)',xticks=range(1,20),title='All 19 windows, record-shared development');ax.legend();ax.grid(alpha=.2);fig.savefig(R/'positions.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
