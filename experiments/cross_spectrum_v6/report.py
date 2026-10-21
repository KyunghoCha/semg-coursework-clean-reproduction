"""Aggregate accuracy and saved-size comparison; no person-level points."""
from semg.paths import RUN_ROOT
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=RUN_ROOT/'results/cross_spectrum_v6'
assert json.loads((R/'verification.json').read_text())['models_restored']==9
scores=pd.read_csv(R/'scores.csv');positions=pd.read_csv(R/'positions.csv')
labels={'global8_3200':'8 targets / 3,200 rounds','subclass64_400':'64 targets / 400 rounds','subclass64_cross400':'64 targets + cross spectra'}
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for name,g in scores.groupby('model'):
    pos=positions[positions.model==name].groupby('start_seconds').accuracy.mean()
    line=axes[0].plot(pos.index,pos*100,'o-',label=labels[name])[0]
    x=g.model_megabytes.mean();y=g.accuracy.mean()*100
    axes[1].scatter(x,y,s=95,color=line.get_color())
    axes[1].annotate(labels[name],(x,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
axes[0].set(xlabel='Window start after marker (s)',ylabel='Window accuracy (%)',ylim=(65,100),title='Same 200 ms signal and all 19 windows')
axes[0].legend(fontsize=8)
axes[1].set(xlabel='Mean saved model + scaler (MB)',ylabel='Development window accuracy (%)',xlim=(32,61),ylim=(80,92),title='Same 25,600 trees; different node counts')
fig.savefig(R/'accuracy_size.png',dpi=170);plt.close(fig)
