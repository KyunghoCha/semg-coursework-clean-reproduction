"""Keep both unsuccessful and selected aggregate forest results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common28 import P,R,digest


def main():
    assert (R/'verification.json').exists();complete=json.loads((R/'complete.json').read_text());selection=json.loads((R/'selection.json').read_text());cost=json.loads((R/'costs.json').read_text());assert digest(P/'cost_raw.json')==cost['raw_timing_sha256'];raw=json.loads((P/'cost_raw.json').read_text())
    for row in cost['rows']:
        v=np.array(raw[row['role']]);assert len(v)==380
        for key,value in dict(median_ms=np.median(v),mean_ms=np.mean(v),p95_ms=np.percentile(v,95)).items():assert abs(row[key]-value)<1e-10
    inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');pooled=pd.read_csv(R/'pooled.csv');pairs=pd.read_csv(R/'paired_changes.csv');fits=pd.read_csv(R/'fits.csv');summary=dict(**complete,fold_use_forest=selection['fold_use_forest'],inner=inner.groupby('role')[['windows','correct']].sum().reset_index().to_dict('records'),outer=pooled.to_dict('records'),fit_seconds_sum=float(fits.fit_seconds.sum()),scaler_fit_seconds_sum=float(fits.scaler_fit_seconds.sum()),cost=cost['rows'],raw_cost_statistics_recomputed=True,input_ms=200,all_19_windows=True,temporal_context=False,external_independent_test=False,**{c:int(pairs[c].sum()) for c in pairs.columns if c!='fold'})
    (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');positions=np.arange(3);width=.35
    for ax,table,title in [(axes[0],inner,'Inner choice; fixed 10% forest'),(axes[1],outer,'Outer development; frozen choices')]:
        roles=['incumbent','candidate'] if ax==axes[0] else ['incumbent','selected']
        for i,role in enumerate(roles):
            rows=table[table.role==role].sort_values('fold');bars=ax.bar(positions+(i-.5)*width,100*rows.correct/rows.windows,width,label=role);ax.bar_label(bars,fmt='%.2f',fontsize=9)
        ax.set_xticks(positions,['Fold 0','Fold 1','Fold 2']);ax.set_ylim(0,105);ax.set_ylabel('8-class window accuracy (%)');ax.set_title(title);ax.legend()
    fig.suptitle('410 features / 64 targets; cost-bounded forest; same current 200 ms');fig.savefig(R/'comparison.png',dpi=150);plt.close(fig);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
