"""Independent probability sums, selection scores, model restoration and OOF coverage."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0, str(SOURCE_ROOT / 'experiments/spectral_controls_v9'))
from design import TWO

P=RUN_ROOT/'private/probability_blend_v12'; R=RUN_ROOT/'results/probability_blend_v12'
V=RUN_ROOT/'private/nested_selection_v10'
assert (R/'complete.json').exists()
inner=pd.read_csv(R/'inner.csv'); outer=pd.read_csv(R/'outer.csv')
assert len(inner)==15 and len(outer)==12
assert not inner.duplicated(['fold','alpha']).any() and not outer.duplicated(['fold','role']).any()
selection=json.loads((R/'selection.json').read_text())
schedule=json.loads((V/'splits.json').read_text())
sources=json.loads((P/'model_sources.json').read_text()); assert len(sources)==18
for source in sources:
    assert hashlib.sha256((RUN_ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                     np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                     np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
old=json.loads((RUN_ROOT/'private/splits.json').read_text()); dev=set(old['train']); held=set(old['test'])


def choose(rows):
    ranking=[]
    for row in rows:
        a=float(row['alpha']); one=a==0 or a==1
        ranking.append((-int(row['correct']),1 if one else 2,410 if a==1 else 905,abs(a-.5),a))
    return sorted(ranking)[0][-1]


def marginal(prob):
    return np.stack([prob[:,c::8].astype(float).sum(axis=1) for c in range(8)],axis=1)


def check(truth,prob,row):
    pred=prob.argmax(axis=1)
    assert int((truth==pred).sum())==row['correct']
    for key,value in manual_metrics(truth,pred).items():
        assert abs(value-row[key])<1e-12
    normalized=prob/prob.sum(axis=1,keepdims=True)
    assert abs(log_loss(truth,normalized,labels=np.arange(8))-row['log_loss'])<1e-12
    onehot=np.equal.outer(truth,np.arange(8)).astype(float)
    assert abs(((normalized-onehot)**2).sum(axis=1).mean()-row['brier_sum'])<1e-12


coverage=[]
for fold, split in enumerate(schedule):
    tr=np.array(split['outer_train']); te=np.array(split['outer_evaluation'])
    assert not set(tr)&set(te) and set(tr)|set(te)==dev and not (set(tr)|set(te))&held
    coverage.extend(te)
    a=np.load(V/f'inner_fine64_{fold}.npz'); b=np.load(V/f'inner_narrow64_{fold}.npz')
    for key in ['truth','trial']:
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a['trial'],np.repeat(split['inner_evaluation'],19))
    fine=marginal(a['probability']); narrow=marginal(b['probability'])
    rows=inner[inner.fold==fold].to_dict('records')
    assert {r['alpha'] for r in rows}=={0,.25,.5,.75,1}
    for row in rows:
        check(a['truth'],(1-row['alpha'])*fine+row['alpha']*narrow,row)
    assert choose(rows)==selection['fold_alphas'][fold]
    probabilities={}; sizes={}
    for name in ['fine64','narrow64']:
        info=[s for s in sources if s['fold']==fold and s['model']==name and s['path'].endswith('.joblib')]
        assert len(info)==1
        path=RUN_ROOT/info[0]['path']; z=np.load(path.with_suffix('.npz'))
        np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19))
        np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
        x=full if name=='fine64' else full[...,np.r_[np.arange(400),TWO]]
        scaler=StandardScaler().fit(x[tr].reshape(-1,x.shape[-1]))
        saved=joblib.load(path)
        np.testing.assert_array_equal(scaler.mean_,saved['scaler'].mean_)
        np.testing.assert_array_equal(scaler.var_,saved['scaler'].var_)
        np.testing.assert_array_equal(saved['model'].classes_,np.arange(64))
        restored=saved['model'].get_booster().predict(DMatrix(saved['scaler'].transform(x[te].reshape(-1,x.shape[-1]))))
        np.testing.assert_array_equal(restored,z['probability'])
        probabilities[name]=marginal(restored); sizes[name]=path.stat().st_size
    for row in outer[outer.fold==fold].to_dict('records'):
        alpha=row['alpha']; expected={'fine':0.,'narrow':1.,'fixed_half':.5,'selected':selection['fold_alphas'][fold]}[row['role']]
        assert alpha==expected
        prob=(1-alpha)*probabilities['fine64']+alpha*probabilities['narrow64']
        z=np.load(P/f"{row['role']}_{fold}.npz")
        np.testing.assert_array_equal(z['probability'],prob)
        np.testing.assert_array_equal(z['prediction'],prob.argmax(axis=1))
        check(np.repeat(y[te],19),prob,row)
        expected_bytes=sizes['fine64'] if alpha==0 else sizes['narrow64'] if alpha==1 else sum(sizes.values())
        assert row['model_bytes']==expected_bytes
assert sorted(coverage)==sorted(dev)
totals=[dict(alpha=a,correct=int(inner[inner.alpha==a].correct.sum())) for a in [0,.25,.5,.75,1]]
assert selection['final_development_alpha']==choose(totals) and not selection['outer_scores_used']
pooled=[]
for role in outer.role.unique():
    zs=[np.load(P/f'{role}_{fold}.npz') for fold in range(3)]
    t=np.concatenate([z['truth'] for z in zs]); p=np.concatenate([z['prediction'] for z in zs])
    pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),**manual_metrics(t,p)))
pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False)
outer.groupby('role').agg(accuracy=('accuracy','mean'),fold_std=('accuracy',lambda a:a.std(ddof=0)),
                          mean_bytes=('model_bytes','mean'),log_loss=('log_loss','mean'),brier_sum=('brier_sum','mean')).to_csv(R/'summary.csv')
(R/'verification.json').write_text(json.dumps(dict(source_artifact_hashes_checked=18,models_restored=6,
    inner_weight_selection_recalculated=True,probability_arithmetic_independently_recalculated=True,
    every_development_trial_evaluated_once=True,heldout_intersection=0,scalers_train_only=True,
    external_independent_data=False),indent=2)+'\n')
print(pd.DataFrame(pooled).to_string(index=False))
