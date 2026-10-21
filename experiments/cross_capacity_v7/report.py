"""Aggregate all three registered development folds."""
from semg.paths import RUN_ROOT
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=RUN_ROOT/'results/cross_capacity_v7'
assert (R/'verification.json').exists()
scores=pd.read_csv(R/'scores.csv');positions=pd.read_csv(R/'positions.csv')
labels={'cross400':'Cross / 400 rounds','cross800':'Cross / 800 rounds','without_high_cross400':'Remove 250–350 Hz relations'}
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
for name,g in scores.groupby('model'):
    p=positions[positions.model==name].groupby('start_seconds').accuracy.mean()
    line=axes[0].plot(p.index,p*100,label=labels[name])[0]
    axes[1].scatter(g.model_megabytes.mean(),g.accuracy.mean()*100,color=line.get_color(),s=80,label=labels[name])
axes[0].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',title='All 19 positions retained')
axes[1].set(xlabel='Mean model + scaler size (MB)',ylabel='Window accuracy (%)',title='Fixed 200 ms input and same folds')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(R/'accuracy_size.png',dpi=170);plt.close(fig)
