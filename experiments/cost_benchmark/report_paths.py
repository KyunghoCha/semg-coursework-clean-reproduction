"""Plot verified API and paired-vectorization costs without new measurements."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/cost_benchmark'
assert (R/'inplace_verification.json').exists() and (R/'fast_timing_verification.json').exists()
a=pd.read_csv(R/'api_comparison.csv');b=pd.read_csv(R/'feature_optimization.csv')
names=['cross64_400_v7','cross64_800_v7','fine64_400_v8'];labels=['Cross / 400','Cross / 800','Fine power / 400'];x=np.arange(3)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for j,(key,label) in enumerate([('original','Original features'),('vectorized','Vectorized features')]):
    v=[b[(b.model==n)&(b.implementation==key)&(b.stage=='total')].iloc[0] for n in names]
    med=np.array([r.median_ms for r in v]);p95=np.array([r.p95_ms for r in v])
    axes[0].bar(x+(j-.5)*.32,med,.32,label=label)
    axes[0].errorbar(x+(j-.5)*.32,med,yerr=[np.zeros(3),p95-med],fmt='none',ecolor='black',capsize=3)
for j,(key,label) in enumerate([('dmatrix','DMatrix'),('inplace','Inplace + transfer')]):
    v=[a[(a.model==n)&(a.device=='cuda')&(a.api==key)].iloc[0] for n in names]
    med=np.array([r.total_median_ms for r in v]);p95=np.array([r.total_p95_ms for r in v])
    axes[1].bar(x+(j-.5)*.32,med,.32,label=label)
    axes[1].errorbar(x+(j-.5)*.32,med,yerr=[np.zeros(3),p95-med],fmt='none',ecolor='black',capsize=3)
axes[0].set(title='Paired CPU extraction + inplace inference',ylabel='Batch 1 total latency (ms)')
axes[1].set(title='GPU API comparison, copies included',ylabel='Batch 1 total latency (ms)')
for ax in axes:ax.set_xticks(x,labels);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Bar: median; whisker: p95. Same saved models, 200 ms signal.')
fig.savefig(R/'inference_paths.png',dpi=170);plt.close(fig)
