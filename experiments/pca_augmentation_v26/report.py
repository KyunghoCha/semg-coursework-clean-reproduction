"""Recompute timing summaries and plot only aggregate fold results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import P,R,digest


def main():
    assert (R/'verification.json').exists();complete=json.loads((R/'complete.json').read_text());choice=json.loads((R/'selection.json').read_text());costfile=R/('costs_isolated.json' if (R/'costs_isolated.json').exists() else 'costs.json');cost=json.loads(costfile.read_text());rawfile=P/('cost_raw'+cost.get('measurement_suffix','')+'.json');assert digest(rawfile)==cost['raw_timing_sha256'];raw=json.loads(rawfile.read_text())
    for r in cost['rows']:
        a=np.array(raw[r['role']+'_'+r['mode']]);assert len(a)==r['count']==380
        for key,value in dict(median_ms=np.median(a),mean_ms=np.mean(a),p95_ms=np.percentile(a,95)).items():assert abs(r[key]-value)<1e-10
    inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');pooled=pd.read_csv(R/'pooled.csv');fits=pd.read_csv(R/'fits.csv');pairs=pd.read_csv(R/'paired_changes.csv');summary=dict(**complete,fold_use_pca=choice['fold_use_pca'],inner=inner.groupby('role')[['windows','correct']].sum().reset_index().to_dict('records'),outer=pooled.to_dict('records'),fit_seconds_sum=float(fits.fit_seconds.sum()),representation_seconds_sum=float(fits.representation_seconds.sum()),model_bytes_sum=int(fits.model_bytes.sum()),cost=cost['rows'],cost_source=costfile.name,raw_cost_statistics_recomputed=True,input_ms=200,temporal_context=False,all_19_windows=True,external_independent_test=False,**{c:int(pairs[c].sum()) for c in pairs.columns if c!='fold'})
    (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');positions=np.arange(3);width=.35
    for ax,table,title in [(axes[0],inner,'Inner choice; all training rows'),(axes[1],outer,'Outer development; frozen choices')]:
        roles=['incumbent','candidate'] if ax==axes[0] else ['incumbent','selected']
        for i,role in enumerate(roles):
            rows=table[table.role==role].sort_values('fold');values=100*rows.correct/rows.windows;bars=ax.bar(positions+(i-.5)*width,values,width,label=role)
            ax.bar_label(bars,fmt='%.2f',fontsize=9)
        ax.set_xticks(positions,['Fold 0','Fold 1','Fold 2']);ax.set_ylim(0,105);ax.set_ylabel('8-class window accuracy (%)');ax.set_title(title);ax.legend()
    fig.suptitle('PCA coordinates appended; same current 200 ms / all 19 windows');fig.savefig(R/'comparison.png',dpi=150);plt.close(fig);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
