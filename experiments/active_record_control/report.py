"""Matched feature comparison; no selection or new model evaluation."""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/active_record_control'
if (R/'matched_feature_summary.csv').exists():raise FileExistsError('Preserve completed comparison')
parts=[];folds=[]
for features,stage in [(25,'active_record_baseline'),(905,'active_record_control')]:
    directory=RUN_ROOT/'results'/stage;assert (directory/'verification.json').exists()
    p=pd.read_csv(directory/'pooled.csv');p.insert(0,'features',features);parts.append(p)
    f=pd.read_csv(directory/'scores.csv');f['features']=features;folds.append(f)
base=json.loads((RUN_ROOT/'private/active_record_baseline/splits.json').read_text())
full=json.loads((RUN_ROOT/'private/active_record_control/splits.json').read_text());assert base==full
summary=pd.concat(parts,ignore_index=True);summary.to_csv(R/'matched_feature_summary.csv',index=False)
byfold=pd.concat(folds,ignore_index=True)
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for j,(features,label,color) in enumerate([(25,'Lecture 25 features','#3377aa'),(905,'Extended 905 features','#dd7744')]):
    for i,protocol in enumerate(['trial_shared','record_disjoint']):
        x=i+(j-.5)*.32
        score=float(summary[(summary.features==features)&(summary.protocol==protocol)].accuracy.iloc[0])*100
        a=byfold[(byfold.features==features)&(byfold.protocol==protocol)].accuracy.to_numpy()*100
        ax.bar(x,score,width=.29,color=color,label=label if i==0 else None)
        ax.scatter(np.repeat(x,len(a)),a,s=16,color='black',zorder=3)
        ax.text(x,max(score,max(a))+2,f'{score:.2f}%',ha='center',fontsize=10)
ax.set(xticks=[0,1],xticklabels=['Disjoint trials, shared recordings','Disjoint recordings'],ylim=(0,102),
       ylabel='All activity-window accuracy (%)',title='Same direct-8 XGB400 / same splits / all 19 windows')
ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);fig.savefig(R/'matched_feature_transfer.png',dpi=170);plt.close(fig)
(R/'matched_comparison_verification.json').write_text(json.dumps(dict(splits_identical=True,features=[25,905],
    same_model_rounds=400,same_training_targets=8,windows_per_protocol_per_representation=29184,
    posthoc_diagnostic=True,main_candidate_selection_changed=False,source_models_retrained=False),indent=2)+'\n')
print(summary.to_string(index=False))
