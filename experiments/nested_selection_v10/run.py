"""Registered inner selection and conditional outer evaluation on reused development data."""
import gc
import hashlib
import json
import sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import unfold,metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/spectral_controls_v9'))
from design import design
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved
from training_progress import TrainingProgress
P=RUN_ROOT/'private/nested_selection_v10';R=RUN_ROOT/'results/nested_selection_v10'
from selection_rule import NAMES,candidate,select


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    resume='--resume' in sys.argv
    if any((R/f'{stage}.csv').exists() for stage in ['inner','outer']) and not resume:raise FileExistsError('Explicit --resume required')
    if resume:snapshot(P)
    done={stage:checked_rows(R/f'{stage}.csv',P,['model','fold'],lambda r,s=stage:f"{s}_{r['model']}_{r['fold']}",lambda r:['.npz','.json','.joblib']) for stage in ['inner','outer']}
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test'])
    y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
    full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
    assert full.shape==(1920,19,905)
    meta=pd.read_csv(RUN_ROOT/'private/trials.csv');subject=meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    schedule=[];manifest=[]
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b]
        ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+fold).split(tr,y[tr]))
        item=dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist())
        assert len(tr)==1024 and len(te)==512 and len(ia)==768 and len(ib)==256
        assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        assert not set(tr[ia])&set(tr[ib]) and set(tr[ia])|set(tr[ib])==set(tr)
        schedule.append(item)
        for role,ids in {**item,'historical_test':sorted(held)}.items():
            manifest.extend(dict(protocol='nested_selection_v10',fold=fold,role=role,trial_id=i) for i in ids)
    write_identical_or_new(P/'splits.json',schedule)
    pd.DataFrame(manifest).to_csv(P/'split_manifest.csv',index=False)
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/nested_selection_v10/PLAN.md').read_bytes()).hexdigest(),candidates=NAMES,outer_seed=20261109,inner_seed_base=20261110,inner_fits=12,outer_fits_range=[3,6],development_trials=1536,historical_test_excluded=384,external_independent_data=False))
    preflight=[]
    for fold,s in enumerate(schedule):
        labels=subject[s['inner_train']]*8+y[s['inner_train']]
        counts=np.bincount(labels,minlength=64);assert len(counts)==64 and np.all(counts>0)
        preflight.append(dict(fold=fold,inner_train_trials=768,inner_selection_trials=256,outer_evaluation_trials=512,learning_targets=64,minimum_inner_trials_per_target=int(counts.min()),maximum_inner_trials_per_target=int(counts.max())))
    write_identical_or_new(R/'preflight.json',dict(folds=preflight,all_targets_present=True,historical_test_excluded=True,features_shape=list(full.shape),new_fits=0))
    if '--prepare-only' in sys.argv:
        print(json.dumps(preflight),flush=True);return
    def fit(name,fold,stage,tr,te):
        key=(name,fold)
        if key in done[stage]:print('RESUME_SKIP',stage,key,flush=True);return done[stage][key]
        stem=f'{stage}_{name}_{fold}';assert not list(P.glob(f'{stem}.*')), 'Unscored artifacts require review'
        x,classes,rounds=candidate(name,full);a,ya,_=unfold(x,y,tr);b,yb,gb=unfold(x,y,te)
        joint=np.repeat(subject[tr]*8+y[tr],19);assert np.array_equal(np.unique(joint),np.arange(64))
        scaler=StandardScaler();aa=scaler.fit_transform(a);bb=scaler.transform(b)
        model=make_model('XGB');model.set_params(n_jobs=6,n_estimators=rounds,callbacks=[TrainingProgress(stem,rounds)])
        t=perf_counter();model.fit(aa,joint if classes==64 else ya);elapsed=perf_counter()-t;model.set_params(callbacks=None)
        config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
        assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==25600
        probability=model.get_booster().predict(DMatrix(bb));np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
        eight=probability.reshape(-1,8,8).sum(axis=1) if classes==64 else probability
        pred=eight.argmax(axis=1);causal=aggregate(pred.reshape(-1,19)).ravel();dest=P/f'{stem}.joblib'
        joblib.dump(dict(model=model,scaler=scaler),dest)
        np.savez(P/f'{stem}.npz',truth=yb,prediction=pred,trial=gb,probability=probability,causal=causal,scaler_mean=scaler.mean_,scaler_var=scaler.var_)
        (P/f'{stem}.json').write_text(json.dumps(config,indent=2))
        row=dict(model=name,fold=fold,stage=stage,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),evaluation_windows=len(b),learning_classes=classes,output_classes=8,trees=25600,rounds=rounds,features=x.shape[-1],fit_seconds=elapsed,model_megabytes=dest.stat().st_size/1e6,correct=int((pred==yb).sum()),causal_accuracy=float((causal==yb).mean()),**metrics(yb,pred))
        path=R/f'{stage}.csv';pd.DataFrame([row]).to_csv(path,mode='a',header=not path.exists(),index=False);print(json.dumps(row),flush=True)
        done[stage][key]=row;del x,a,b,aa,bb,model,scaler;gc.collect();return row
    roles=[];choices=[]
    for fold,s in enumerate(schedule):
        inner=[fit(name,fold,'inner',np.array(s['inner_train']),np.array(s['inner_evaluation'])) for name in NAMES]
        chosen=select(inner);choices.append(chosen)
        write_identical_or_new(R/f'selection_fold_{fold}.json',dict(fold=fold,selected=chosen,source='inner correct window counts only',scores={row['model']:int(row['correct']) for row in inner}))
        # Reference and selected use one fit only when all settings coincide.
        for name in dict.fromkeys([chosen,'fine64']):
            fit(name,fold,'outer',np.array(s['outer_train']),np.array(s['outer_evaluation']))
        for role,name in [('selected',chosen),('reference','fine64')]:roles.append(dict(fold=fold,role=role,model=name,stem=f'outer_{name}_{fold}'))
    pd.DataFrame(roles).to_csv(R/'outer_roles.csv',index=False)
    total=[]
    for name in NAMES:
        rows=[done['inner'][(name,k)] for k in range(3)]
        total.append(dict(model=name,correct=sum(r['correct'] for r in rows),features=rows[0]['features'],rounds=rows[0]['rounds']))
    write_identical_or_new(R/'selection.json',dict(outer_choices=choices,final_development_candidate=select(total),inner_correct_totals=total,outer_scores_used=False,historical_test_used=False))
    if resume:verify_preserved(P)
    write_identical_or_new(R/'complete.json',dict(inner_fits=12,outer_fits=len(done['outer']),new_fits=12+len(done['outer']),historical_test_evaluations=0))


if __name__=='__main__':main()
