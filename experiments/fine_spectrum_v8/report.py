"""Within-fold reference/capacity/fine-power comparison; aggregate-only figures."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/fine_spectrum_v8';V=RUN_ROOT/'results/cross_capacity_v7'
assert (R/'verification.json').exists()
a=pd.read_csv(V/'pooled.csv');b=pd.read_csv(R/'pooled.csv')
combined=pd.concat([a,b],ignore_index=True);combined.to_csv(R/'comparison.csv',index=False)
scores=pd.concat([pd.read_csv(V/'scores.csv'),pd.read_csv(R/'scores.csv')],ignore_index=True)
summary=scores.groupby('model').agg(accuracy=('accuracy','mean'),std=('accuracy',lambda x:x.std(ddof=0)),
    model_mb=('model_megabytes','mean'),fit_seconds=('fit_seconds','mean')).reset_index()
summary.to_csv(R/'comparison_summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
positions=pd.concat([pd.read_csv(V/'positions.csv'),pd.read_csv(R/'positions.csv')])
labels={'cross400':'Cross / 400','cross800':'Cross / 800','without_high_cross400':'Remove high-band relations','fine400':'Cross + fine power / 400'}
for i,name in enumerate(labels):
    p=positions[positions.model==name].groupby('start_seconds').accuracy.mean()
    axes[0].plot(p.index,p.values*100,label=labels[name],lw=1.8)
    row=summary[summary.model==name].iloc[0]
    axes[1].scatter(row.model_mb,row.accuracy*100,s=80,label=labels[name],color=f'C{i}')
axes[0].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',title='All 19 positions retained')
axes[1].set(xlabel='Mean saved model + scaler (MB)',ylabel='Window accuracy (%)',title='Same development folds and 200 ms input')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(R/'accuracy_size.png',dpi=170);plt.close(fig)
print(summary.to_string(index=False))
