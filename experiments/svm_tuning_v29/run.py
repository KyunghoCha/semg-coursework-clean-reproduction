"""Fit only preregistered alternatives; freeze inner choices before outer fits."""
import json
from datetime import datetime, timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.evaluation import metrics
from common29 import (P, R, inputs, source_paths, load, snapshot, check_hashes, digest,
                      betas, combine, configure_audit, make_model, trial_cv, model_path,
                      CANDIDATES, SIZE_CAP, choose, EXPERIMENT)


def write(path, obj):
    path.open('x').write(json.dumps(obj, indent=2)+'\n')


def main():
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    if (R/'registration.json').exists(): raise FileExistsError('Inspect existing outcomes before retry')
    x, y, target, schedule = inputs(); original = json.loads((RUN_ROOT/'private/splits.json').read_text())
    dev, held = set(original['train']), set(original['test']); bs = betas()
    write(P/'splits.json', schedule)
    hashes = snapshot([RUN_ROOT/'private/splits.json', RUN_ROOT/'private/nested_selection_v10/splits.json',
                       RUN_ROOT/'results/mixture_weights_v25/selection.json'])
    for k in range(3):
        paths = source_paths('inner', k); hashes.update(snapshot(paths+[p.with_suffix('.joblib') for p in paths]))
    write(P/'source_hashes_inner.json', hashes)
    write(R/'registration.json', dict(utc=datetime.now(timezone.utc).isoformat(),
          plan_sha256=digest(SOURCE_ROOT/'experiments'/EXPERIMENT/'PLAN.md'), candidates=CANDIDATES,
          size_cap=SIZE_CAP, fold_betas=bs, baseline_head='7690b21bf3aeec9e3d24c8b551920277e4a90a09',
          execution_source_hashes={str(path.relative_to(SOURCE_ROOT)):digest(path) for path in [
              SOURCE_ROOT/'experiments/svm_tuning_v29/run.py',SOURCE_ROOT/'experiments/svm_tuning_v29/common29.py',
              SOURCE_ROOT/'experiments/svm_tuning_v29/tuning_rules.py',SOURCE_ROOT/'experiments/svm410_v20/audited_svm.py']+([SOURCE_ROOT/'experiments/svm_metric_v31/covariance_metric.py'] if EXPERIMENT=='svm_metric_v31' else [])},
          representation=('full-dimensional LedoitWolf covariance metric' if EXPERIMENT=='svm_metric_v31' else 'train-only StandardScaler'),
          signal_ms=200, stride_ms=100, windows_per_trial=19, development_trials=1536,
          inner_models=3*len(CANDIDATES), maximum_outer_models=3, four_base_fits_per_model=True,
          internal_calibration_trial_grouped=True, temporal_context=False))
    fits = []; support = []

    def fit(stage, k, name):
        s = schedule[k]; tr = np.array(s[stage+'_train']); te = np.array(s[stage+'_evaluation'])
        assert not set(tr)&set(te) and (set(tr)|set(te))<=dev and not (set(tr)|set(te))&held
        stem = f'{stage}_{name}_{k}'; a = x[tr].reshape(-1,410); b = x[te].reshape(-1,410)
        labels = np.repeat(target[tr],19); np.testing.assert_array_equal(np.unique(labels),np.arange(64))
        cv = trial_cv(target[tr], (20261123 if stage=='inner' else 20261133)+k)
        for j,(ca,cb) in enumerate(cv):
            assert not set(ca//19)&set(cb//19)
            np.testing.assert_array_equal(np.unique(labels[cb]),np.arange(64))
            counts=np.bincount(labels[ca],minlength=64)
            support.append(dict(stage=stage,fold=k,candidate=name,calibration_fold=j,train_trials=len(ca)//19,
                                calibration_trials=len(cb)//19,min_trials_per_target=int(counts.min()//19),
                                max_trials_per_target=int(counts.max()//19)))
        configure_audit(P/stem); m=make_model(cv)
        if EXPERIMENT=='svm_metric_v31':
            from covariance_metric import ShrinkageWhitening
            m.estimator.steps.insert(1,('covariance',ShrinkageWhitening()))
        m.estimator.set_params(**{'svc__'+key:value for key,value in CANDIDATES[name].items()})
        print('CALIBRATED_START',stem,datetime.now(timezone.utc).isoformat(),flush=True)
        start=perf_counter(); m.fit(a,labels); seconds=perf_counter()-start
        logs=[json.loads(p.read_text()) for p in sorted((P/stem).glob('base_*.json'))]
        assert len(logs)==4 and logs[-1]['windows']==len(a) and all(t['fit_status']==0 for t in logs)
        np.testing.assert_array_equal(m.classes_,np.arange(64)); q=m.predict_proba(b)
        np.testing.assert_allclose(q.sum(1),1,atol=1e-10)
        truth=np.repeat(y[te],19); pred=q.reshape(-1,8,8).sum(1).argmax(1)
        path=model_path(stage,k,name); assert not path.exists(); joblib.dump(m,path)
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,prediction=pred)
        row=dict(stage=stage,fold=k,candidate=name,utc=datetime.now(timezone.utc).isoformat(),
                 train_trials=len(tr),evaluation_trials=len(te),evaluation_windows=len(truth),
                 calibrated_model_fit_seconds=seconds,base_svc_fit_seconds=sum(t['fit_seconds'] for t in logs),
                 base_svc_fits=4,model_bytes=path.stat().st_size,support_vectors=logs[-1]['support_vectors'],
                 max_pair_iterations=logs[-1]['maximum_pair_iterations'],all_fit_status_zero=True,
                 correct=int((pred==truth).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'fits.csv',mode='a',header=not(R/'fits.csv').exists(),index=False)
        fits.append(row);print('CALIBRATED_DONE',json.dumps(row),flush=True);return q,row

    inner=[]; choices=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']); qs=load('inner',k,ids,y); truth=np.repeat(y[ids],19)
        old_svc=joblib.load(model_path('inner',k,'incumbent')).calibrated_classifiers_[0].estimator.named_steps['svc']
        q=combine(*qs,bs[k]); pred=q.argmax(1)
        rows=[dict(fold=k,candidate='incumbent',correct=int((pred==truth).sum()),
                   model_bytes=model_path('inner',k,'incumbent').stat().st_size,support_vectors=len(old_svc.support_),
                   **metrics(truth,pred))]
        for name in CANDIDATES:
            prob,row=fit('inner',k,name); mix=combine(qs[0],qs[1],prob,bs[k]); pred=mix.argmax(1)
            rows.append(dict(fold=k,candidate=name,correct=int((pred==truth).sum()),model_bytes=row['model_bytes'],
                             support_vectors=row['support_vectors'],**metrics(truth,pred)))
        inner.extend(rows);choices.append(choose(rows));pd.DataFrame(inner).to_csv(R/'inner.csv',index=False)
        print('INNER_FOLD_COMPLETE',k,json.dumps(rows),flush=True)
    write(R/'selection.json',dict(utc=datetime.now(timezone.utc).isoformat(),fold_candidates=choices,
                                 fold_betas=bs,outer_scores_used=False,rule='correct, size cap, incumbent tie priority'))
    selection_hash=digest(R/'selection.json');print('FROZEN_SELECTION',choices,selection_hash,flush=True)
    out=[]
    for k,(s,name) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);qs=load('outer',k,ids,y);paths=source_paths('outer',k)
        hashes.update(snapshot(paths+[p.with_suffix('.joblib') for p in paths]));truth=np.repeat(y[ids],19)
        old=combine(*qs,bs[k]); expected=np.load(RUN_ROOT/f'private/mixture_weights_v25/outer_selected_{k}.npz')
        np.testing.assert_array_equal(old.argmax(1),expected['prediction'])
        candidate=old;applied=name;size_pass=True
        if name!='incumbent':
            prob,row=fit('outer',k,name);candidate=combine(qs[0],qs[1],prob,bs[k])
            size_pass=row['model_bytes']<=SIZE_CAP*model_path('outer',k,'incumbent').stat().st_size
            if not size_pass:applied='incumbent'
        selected=candidate if size_pass else old
        for role,q in [('incumbent',old),('candidate',candidate),('selected',selected)]:
            pred=q.argmax(1);np.savez(P/f'outer_{role}_{k}.npz',trial=np.repeat(ids,19),truth=truth,probability=q,prediction=pred)
            out.append(dict(fold=k,role=role,inner_choice=name,applied_candidate=applied,outer_size_cap_pass=size_pass,
                            windows=len(truth),correct=int((pred==truth).sum()),**metrics(truth,pred)))
        assert digest(R/'selection.json')==selection_hash
        pd.DataFrame(out).to_csv(R/'outer.csv',index=False)
        print('OUTER_FOLD_COMPLETE',json.dumps(out[-1]),flush=True)
    pd.DataFrame(support).to_csv(R/'calibration_support.csv',index=False);check_hashes(hashes)
    write(P/'source_hashes.json',hashes)
    write(R/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),calibrated_models=len(fits),
          base_svc_fits=4*len(fits),inner_models=3*len(CANDIDATES),outer_models=len(fits)-3*len(CANDIDATES),
          selection_sha256=selection_hash,historical_test_evaluations=0))


if __name__=='__main__':
    with threadpool_limits(limits=6): main()
