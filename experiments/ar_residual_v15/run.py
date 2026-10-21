"""Predeclared inner-only AR choice, then fixed direct-8 record controls."""
import gc, hashlib, json, sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import metrics
sys.path.insert(0, str(SOURCE_ROOT/'experiments'))
from training_progress import TrainingProgress
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved
sys.path.insert(0, str(SOURCE_ROOT/'experiments/spectral_controls_v9'))
from design import TWO
P = RUN_ROOT/'private/ar_residual_v15'
R = RUN_ROOT/'results/ar_residual_v15'


def source_base(stage,fold):
    path=RUN_ROOT/f'private/nested_selection_v10/{stage}_narrow64_{fold}.npz'
    if not path.exists():path=RUN_ROOT/f'private/probability_blend_v12/{stage}_narrow64_{fold}.npz'
    return path


def features(name):
    ar = np.load(P/'features.npy')
    if name.startswith('td'):
        base = np.load(RUN_ROOT/'private/cache/baseline.npz')['X']
    else:
        base = np.concatenate([
            np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
            np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
            np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')[:, :, TWO-400]], axis=2)
        assert base.shape == (1920,19,410)
    return np.concatenate([base, ar if name.endswith('_res') else ar[:,:,:30]], axis=2)


def main():
    P.mkdir(exist_ok=True); R.mkdir(exist_ok=True)
    resume='--resume' in sys.argv
    if ((R/'registration.json').exists() or (R/'scores.csv').exists()) and not resume:
        raise FileExistsError('Explicit --resume requires checked completed artifacts')
    if (R/'complete.json').exists():raise FileExistsError('Run already complete')
    if resume:snapshot(P)
    done=checked_rows(R/'scores.csv',P,['stage','model','fold'],lambda r:f"{r['stage']}_{r['model']}_{r['fold']}",lambda r:['.npz','.joblib','.json'])
    assert (R/'features.json').exists()
    old = json.loads((RUN_ROOT/'private/splits.json').read_text())
    dev = set(old['train']); held = set(old['test'])
    y = np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
    meta = pd.read_csv(RUN_ROOT/'private/trials.csv')
    subject = meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    nested = json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text())
    controls = json.loads((RUN_ROOT/'private/active_record_baseline/splits.json').read_text())
    write_identical_or_new(P/'splits.json',dict(nested=nested,controls=controls))
    plan = SOURCE_ROOT/'experiments/ar_residual_v15/PLAN.md'
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        development_trials=1536,excluded_historical_test=384,full_feature_trials=1920,
        inner_fits=6,outer_fits_max=3,control_fits=16,signal_ms=200,windows_per_trial=19,
        primary_metric='window accuracy',external_independent_data=False))
    supports=[]; allrows=list(done.values())

    def fit(name, fold, stage, tr, te, classes):
        tr=np.array(tr); te=np.array(te); stem=f'{stage}_{name}_{fold}'
        assert not set(tr)&set(te) and (set(tr)|set(te)) <= dev and not(set(tr)|set(te))&held
        if stage!='inner':assert set(tr)|set(te)==dev
        common=len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording))
        assert common==(0 if stage=='record_disjoint' else meta.iloc[te].recording.nunique())
        if stage=='inner':assert meta.iloc[tr].recording.nunique()==64
        labels=subject*8+y if classes==64 else y
        for role, ids in [('train',tr),('evaluation',te)]:
            counts=np.bincount(labels[ids],minlength=classes)
            assert len(np.unique(y[ids]))==8
            if role=='train':assert np.all(counts>0)
            supports.append(dict(stem=stem,role=role,classes=classes,present_targets=int(np.count_nonzero(counts)),min_trials=int(counts.min()),max_trials=int(counts.max()),total_trials=len(ids),record_intersection=common))
        key=(stage,name,fold)
        if key in done:
            print('RESUME_SKIP',stem,flush=True);return done[key]
        assert not list(P.glob(stem+'.*')), 'Unscored artifacts require review'
        x=features(name); width=x.shape[-1]
        scaler=StandardScaler();a=scaler.fit_transform(x[tr].reshape(-1,width));b=scaler.transform(x[te].reshape(-1,width))
        model=make_model('XGB');model.set_params(n_jobs=6,callbacks=[TrainingProgress('ar15_'+stem,400)])
        start=perf_counter();model.fit(a,np.repeat(labels[tr],19));elapsed=perf_counter()-start;model.set_params(callbacks=None)
        cfg=json.loads(model.get_booster().save_config());assert cfg['learner']['generic_param']['device']=='cuda:0'
        assert int(cfg['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==400*classes
        np.testing.assert_array_equal(model.classes_,np.arange(classes))
        q=model.get_booster().predict(DMatrix(b,nthread=6));q8=q.reshape(-1,8,8).sum(axis=1) if classes==64 else q
        pred=q8.argmax(axis=1);truth=np.repeat(y[te],19)
        joblib.dump(dict(model=model,scaler=scaler),P/f'{stem}.joblib')
        np.savez(P/f'{stem}.npz',truth=truth,prediction=pred,probability=q,trial=np.repeat(te,19))
        (P/f'{stem}.json').open('x').write(json.dumps(cfg,indent=2))
        row=dict(stage=stage,model=name,fold=fold,train_trials=len(tr),evaluation_trials=len(te),evaluation_windows=len(truth),features=width,classes=classes,
                 trees=classes*400,rounds=400,correct=int((truth==pred).sum()),fit_seconds=elapsed,model_bytes=(P/f'{stem}.joblib').stat().st_size,**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False)
        allrows.append(row);print(json.dumps(row),flush=True)
        del x,a,b,model,scaler;gc.collect();return row

    inner=[]
    for k,s in enumerate(nested):
        for name in ['narrow_ar','narrow_ar_res']:
            inner.append(fit(name,k,'inner',s['inner_train'],s['inner_evaluation'],64))
    baseline=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/inner.csv')
    choices=[];comparison=[]
    for k in range(3):
        b=baseline[(baseline.model=='narrow64')&(baseline.fold==k)].iloc[0]
        candidates=[dict(model='narrow64',features=410,correct=int(b.correct))]+[r for r in inner if r['fold']==k]
        chosen=min(candidates,key=lambda r:(-r['correct'],r['features']))['model'];choices.append(chosen)
        comparison.extend(dict(fold=k,model=r['model'],correct=r['correct'],features=r['features']) for r in candidates)
    (R/'selection.json').open('x').write(json.dumps(dict(choices=choices,comparisons=comparison,source='inner validation correct counts only',outer_scores_used=False),indent=2)+'\n')
    print('FROZEN_SELECTION',choices,flush=True)
    for k,(name,s) in enumerate(zip(choices,nested)):
        if name!='narrow64':fit(name,k,'outer',s['outer_train'],s['outer_evaluation'],64)
    for name in ['td_ar','td_ar_res']:
        for s in controls:fit(name,s['fold'],s['protocol'],s['train'],s['evaluation'],8)
    if resume:verify_preserved(P)
    pd.DataFrame(supports).to_csv(R/'class_support.csv',index=False)
    (R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(allrows),inner_fits=6,outer_fits=sum(r['stage']=='outer' for r in allrows),control_fits=16,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
