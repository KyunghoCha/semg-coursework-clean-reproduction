"""Export ID-based split manifests and quantify cross-protocol reuse explicitly."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from semg.evaluation import folds
from semg.paths import RUN_ROOT

root=RUN_ROOT
meta=pd.read_csv(root/'private/trials.csv')
split=json.loads((root/'private/splits.json').read_text())
rows=[]

def add(protocol,fold,role,indices):
    for i in indices:
        rows.append(dict(protocol=protocol,fold=fold,role=role,global_index=int(i),
                         trial_id=meta.trial_id.iloc[i],label=int(meta.label.iloc[i])))

for role,key in [('train','train'),('test','test')]:add('holdout',0,role,split[key])
for k,s in enumerate(split['folds']):
    add('baseline_cv',k,'train',s['train']);add('baseline_cv',k,'validation',s['test'])
outer=np.array(split['train'])
for k,(a,b) in enumerate(folds(meta.label.to_numpy()[outer])):
    add('development',k,'train',outer[a]);add('development',k,'validation',outer[b])
manifest=pd.DataFrame(rows)
manifest.to_csv(root/'private/split_manifest.csv',index=False)
checks=[];test=set(split['test'])
for (protocol,k),g in manifest.groupby(['protocol','fold']):
    a=set(g[g.role=='train'].global_index)
    b=set(g[g.role!='train'].global_index)
    assert not a&b
    if protocol=='development':assert not ((a|b)&test)
    checks.append(dict(protocol=protocol,fold=int(k),train=len(a),evaluation=len(b),
                       train_eval_intersection=len(a&b),
                       train_intersection_holdout_test=len(a&test),
                       evaluation_intersection_holdout_test=len(b&test)))
pd.DataFrame(checks).to_csv(root/'results/split_checks.csv',index=False)
print(pd.DataFrame(checks).to_string(index=False))
print('manifest_sha256',hashlib.sha256((root/'private/split_manifest.csv').read_bytes()).hexdigest())
train_uses=Counter();validation_uses=Counter()
for s in split['folds']:
    train_uses.update(test&set(s['train']))
    validation_uses.update(test&set(s['test']))
exposure=dict(holdout_test_trials=len(test),
              prior_baseline_cv_training_unique_trials=len(train_uses),
              training_fold_count_per_trial=dict(Counter(train_uses.values())),
              validation_fold_count_per_trial=dict(Counter(validation_uses.values())),
              development_intersection_holdout_test=0,
              interpretation='fixed split of previously used data; not unseen external evaluation')
(root/'results/holdout_exposure.json').write_text(json.dumps(exposure,indent=2)+'\n')
