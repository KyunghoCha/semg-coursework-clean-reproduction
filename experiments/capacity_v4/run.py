"""Feature-informed capacity and kernel comparisons, selected within nested development folds."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import gc
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.svm import SVC
from semg.evaluation import metrics,unfold
from semg.models import make_model,predict
ROOT=RUN_ROOT;P=ROOT/'private/capacity_v4';R=ROOT/'results/capacity_v4'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved
NAMES=['XGB_rich800','XGB_rich1600','XGB_noTime800','SVM_rich_standard','SVM_rich_quantile']


def build_model(name):
    if name.startswith('XGB'):
        model=make_model('XGB');model.set_params(n_estimators=1600 if name=='XGB_rich1600' else 800,n_jobs=6);return model
    return SVC(C=100,kernel='rbf',gamma='scale',cache_size=1024,probability=False,decision_function_shape='ovr')


def fit(name,x,y,tr,te,phase,outer,inner=-1):
    assert not set(tr)&set(te) and not (set(tr)|set(te))&HELD
    a,ya,ga=unfold(x,y,tr);b,yb,gb=unfold(x,y,te)
    scaler=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=None,random_state=42) if name=='SVM_rich_quantile' else StandardScaler()
    t=perf_counter();a=scaler.fit_transform(a);model=build_model(name);model.fit(a,ya);elapsed=perf_counter()-t
    b=scaler.transform(b);p=predict(model,b);tp=predict(model,a)
    if name.startswith('XGB'):
        config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
        assert model.get_booster().num_boosted_rounds()==(1600 if name=='XGB_rich1600' else 800)
    else:
        assert model.fit_status_==0;config=dict(parameters=model.get_params(),support_vectors=int(len(model.support_vectors_)),iterations=model.n_iter_.tolist())
    cp=aggregate(p.reshape(-1,19)).ravel()
    row=dict(candidate=name,phase=phase,outer_fold=outer,inner_fold=inner,train_trials=len(tr),evaluation_trials=len(te),
        train_windows=len(a),evaluation_windows=len(b),features=x.shape[-1],fit_seconds=elapsed,train_accuracy=float((tp==ya).mean()),
        causal_accuracy=float((cp==yb).mean()),**metrics(yb,p))
    stem=f'{phase}_{outer}_{inner}_{name}'
    np.savez(P/f'{stem}.npz',truth=yb,prediction=p,trial=gb,causal=cp)
    np.savez(P/f'{stem}_scaler.npz',**{k:getattr(scaler,k) for k in ['mean_','var_','scale_','quantiles_','references_'] if hasattr(scaler,k)})
    (P/f'{stem}.json').write_text(json.dumps(dict(model=config,scaler=scaler.get_params(),scaler_n=len(a)),indent=2))
    if phase=='outer':joblib.dump(dict(scaler=scaler,model=model),P/f'{stem}.joblib')
    target=R/f'{phase}.csv';pd.DataFrame([row]).to_csv(target,mode='a',header=not target.exists(),index=False)
    print(json.dumps(row),flush=True);del a,b,model,scaler;gc.collect();return row


if __name__=='__main__':
    resume='--resume' in sys.argv
    if not resume and ((R/'inner.csv').exists() or (R/'outer.csv').exists()):raise FileExistsError('Preserve the registered run; explicitly use --resume')
    done={}
    if resume:
        snapshot(P)
        for phase in ['inner','outer']:
            done.update(checked_rows(R/f'{phase}.csv',P,['phase','outer_fold','inner_fold','candidate'],
                lambda r:f"{r['phase']}_{r['outer_fold']}_{r['inner_fold']}_{r['candidate']}",
                lambda r:['.npz','_scaler.npz','.json']+(['.joblib'] if r['phase']=='outer' else [])))
    split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);HELD=set(split['test'])
    y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y'];f=np.load(ROOT/'private/diagnostic_v2/features.npz',allow_pickle=False)
    rich=f['rich'];no_time=np.concatenate([rich[:,:,:25],rich[:,:,110:]],axis=2)
    np.testing.assert_array_equal(no_time,np.concatenate([f['relation'],f['spectrum'][:,:,25:]],axis=2))
    schedule=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261024).split(dev,y[dev])):
        tr,te=dev[a],dev[b];inside=[]
        for c,d in StratifiedKFold(3,shuffle=True,random_state=20261025+k).split(tr,y[tr]):inside.append(dict(train=tr[c].tolist(),validation=tr[d].tolist()))
        schedule.append(dict(train=tr.tolist(),evaluation=te.tolist(),inner=inside))
    write_identical_or_new(P/'splits.json',schedule)
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/capacity_v4/PLAN.md').read_bytes()).hexdigest(),
        candidates=NAMES,development_trials=1536,holdout_excluded=384,outer_folds=3,inner_folds=3,history_windows=5,
        selection='maximum inner mean independent-window accuracy; list order breaks ties',historically_unseen_data=False))
    selected=[]
    for k,s in enumerate(schedule):
        rows=[]
        for name in NAMES:
            x=no_time if name=='XGB_noTime800' else rich
            for j,z in enumerate(s['inner']):
                key=('inner',k,j,name)
                if key in done: rows.append(done[key]);print('RESUME_SKIP',key,flush=True)
                else:
                    assert not list(P.glob(f'inner_{k}_{j}_{name}.*')), 'Unscored artifacts require review'
                    rows.append(fit(name,x,y,np.array(z['train']),np.array(z['validation']),'inner',k,j))
        means=pd.DataFrame(rows).groupby('candidate').accuracy.mean();winner=max(NAMES,key=lambda n:means[n])
        selected.append(dict(outer_fold=k,selected=winner,inner_means=means.to_dict()));(R/'selections.json').write_text(json.dumps(selected,indent=2))
        print('SELECTED',json.dumps(selected[-1]),flush=True)
        for name in (['XGB_rich800',winner] if winner!='XGB_rich800' else [winner]):
            if ('outer',k,-1,name) in done: print('RESUME_SKIP',('outer',k,-1,name),flush=True)
            else:
                assert not list(P.glob(f'outer_{k}_-1_{name}.*')), 'Unscored artifacts require review'
                fit(name,no_time if name=='XGB_noTime800' else rich,y,np.array(s['train']),np.array(s['evaluation']),'outer',k)
    if resume:verify_preserved(P)
    (R/'complete.json').write_text(json.dumps(dict(inner_fits=45,outer_folds_complete=3,selected=selected),indent=2))
