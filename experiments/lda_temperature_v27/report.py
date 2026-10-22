"""Aggregate the fixed temperature experiment and verify raw timing summaries."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sources27 import P,R,digest


def main():
    assert (R/'verification.json').exists();complete=json.loads((R/'complete.json').read_text());selection=json.loads((R/'selection.json').read_text());cost=json.loads((R/'costs.json').read_text());assert digest(P/'cost_raw.json')==cost['raw_timing_sha256'];raw=json.loads((P/'cost_raw.json').read_text())
    for row in cost['rows']:
        a=np.array(raw[row['role']]);assert len(a)==row['windows']==380
        for key,value in dict(median_ms=np.median(a),mean_ms=np.mean(a),p95_ms=np.percentile(a,95)).items():assert abs(row[key]-value)<1e-10
    inner=pd.read_csv(R/'inner.csv');pooled=pd.read_csv(R/'pooled.csv');pairs=pd.read_csv(R/'paired_changes.csv');summary=dict(**complete,fold_temperatures=selection['fold_temperatures'],inner=inner.to_dict('records'),outer=pooled.to_dict('records'),cost=cost['rows'],raw_cost_statistics_recomputed=True,input_ms=200,temporal_context=False,all_19_windows=True,external_independent_test=False,**{c:int(pairs[c].sum()) for c in pairs.columns if c!='fold'})
    (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k in range(3):
        rows=inner[inner.fold==k];axes[0].plot(rows.temperature,rows.accuracy*100,'o-',label=f'Inner fold {k}')
        chosen=rows[rows.temperature==selection['fold_temperatures'][k]].iloc[0];axes[0].scatter([chosen.temperature],[chosen.accuracy*100],marker='*',s=140,edgecolor='black',zorder=4)
    axes[0].set_xlabel('LDA temperature');axes[0].set_ylabel('8-class window accuracy (%)');axes[0].set_title('Inner choice only');axes[0].legend()
    values=pooled.accuracy*100;bars=axes[1].bar(['Fixed v25','Inner-selected T'],values);axes[1].bar_label(bars,fmt='%.4f');axes[1].set_ylim(0,105);axes[1].set_ylabel('8-class window accuracy (%)');axes[1].set_title('Outer development; all 29,184 windows');fig.suptitle('LDA probability softening; same current 200 ms');fig.savefig(R/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='inner'},indent=2))


if __name__=='__main__':main()
