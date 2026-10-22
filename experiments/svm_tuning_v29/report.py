"""Aggregate verified scores and raw cost measurements; preserve failed settings."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common29 import P,R,digest,CANDIDATES,EXPERIMENT


def main():
    assert (R/'verification.json').exists();cost=json.loads((R/'costs.json').read_text())
    assert digest(P/'cost_raw.json')==cost['raw_timing_sha256'];raw=json.loads((P/'cost_raw.json').read_text())
    for row in cost['rows']:
        vals=np.array(raw[row['candidate']]);assert len(vals)==380
        for key,value in dict(median_ms=np.median(vals),mean_ms=np.mean(vals),p95_ms=np.percentile(vals,95)).items():assert abs(row[key]-value)<1e-10
    inner=pd.read_csv(R/'inner.csv');out=pd.read_csv(R/'outer.csv');fits=pd.read_csv(R/'fits.csv')
    summary=json.loads((R/'summary.json').read_text());complete=json.loads((R/'complete.json').read_text())
    summary.update(complete=complete,inner_totals=inner.groupby('candidate').correct.sum().to_dict(),
                   calibrated_wrapper_fit_seconds_sum=float(fits.calibrated_model_fit_seconds.sum()),
                   base_svc_fit_seconds_sum=float(fits.base_svc_fit_seconds.sum()),cost=cost['rows'],
                   raw_cost_statistics_recomputed=True,all_19_current_200ms_windows=True,external_independent_test=False)

    if (R/'report.json').exists():assert json.loads((R/'report.json').read_text())==summary
    else:(R/'report.json').open('x').write(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');positions=np.arange(3);width=.24
    base=inner[inner.candidate=='incumbent'].sort_values('fold').correct.to_numpy()
    for j,name in enumerate(CANDIDATES):
        r=inner[inner.candidate==name].sort_values('fold');delta=r.correct.to_numpy()-base
        bars=ax[0].bar(positions+(j-(len(CANDIDATES)-1)/2)*width,delta,width,label=name);ax[0].bar_label(bars,fmt='%+.0f',padding=3)
    ax[0].set_title('Inner comparison; frozen mixture weights');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=10)
    old=out[out.role=='incumbent'].sort_values('fold').correct.to_numpy();new=out[out.role=='selected'].sort_values('fold').correct.to_numpy()
    bars=ax[1].bar(positions,new-old,.5,color='#3979a9');ax[1].bar_label(bars,fmt='%+.0f',padding=3)
    ax[1].set_title('Outer development; inner-selected settings' if (fits.stage=='outer').any() else 'Outer unchanged: v25 reused; no candidate fit')
    for a in ax:
        a.axhline(0,color='black',linewidth=.8);a.set_xticks(positions,['Fold 0','Fold 1','Fold 2']);a.set_ylabel('Change in correct windows versus v25');
        heights=[p.get_height() for p in a.patches];low=min([0,*heights]);high=max([0,*heights]);pad=max(1.,(high-low)*.25)
        a.set_ylim(low-pad,high+pad)
        from matplotlib.ticker import MaxNLocator
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
        a.title.set_fontsize(12)
    fig.suptitle(EXPERIMENT.rsplit('_',1)[-1]+': current 200 ms / 410 features / all 19 windows per trial',fontsize=14)
    fig.savefig(R/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
