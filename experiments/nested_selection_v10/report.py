import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/nested_selection_v10';assert (R/'verification.json').exists()
i=pd.read_csv(R/'inner.csv');i.groupby('model').agg(accuracy=('accuracy','mean'),std=('accuracy',lambda a:a.std(ddof=0)),fits=('accuracy','count'),mean_fit_seconds=('fit_seconds','mean'),mean_model_mb=('model_megabytes','mean')).to_csv(R/'inner_summary.csv')
o=pd.read_csv(R/'outer_by_fold.csv');o.groupby('role').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),folds='count').to_csv(R/'outer_summary.csv')
p=pd.read_csv(R/'positions.csv');fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
for role,label in [('reference','Fixed fine-power model'),('selected','Inner-selected model')]:
    rows=o[o.role==role];axes[0].plot(rows.fold,rows.accuracy*100,'o-',label=label)
    pos=p[p.role==role].groupby('start_seconds').accuracy.mean();axes[1].plot(pos.index,pos*100,label=label)
axes[0].set(xlabel='Outer fold',ylabel='Window accuracy (%)',title='Selection uses inner data only',xticks=[0,1,2])
axes[1].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',title='All 19 positions retained')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(R/'outer_comparison.png',dpi=170);plt.close(fig)
