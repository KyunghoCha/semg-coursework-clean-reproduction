"""Fixed activity-window counterpart to the verified early-window record control."""
import argparse,gc,hashlib,json,sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.models import make_model
from semg.evaluation import metrics
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from training_progress import TrainingProgress
sys.path.insert(0,str(SOURCE_ROOT/'experiments/record_reference_control'))
from record_splits import schedules
parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--baseline',action='store_true');group.add_argument('--gain',action='store_true');args=parser.parse_args();BASELINE=args.baseline;GAIN=args.gain
NAME='gain_invariance_v14' if GAIN else 'active_record_baseline' if BASELINE else 'active_record_control';WIDTH=25 if BASELINE or GAIN else 905
P=RUN_ROOT/'private'/NAME;R=RUN_ROOT/'results'/NAME


def main():
    assert (RUN_ROOT/'results/forest_transfer_v13/verification.json').exists()
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'scores.csv').exists() or (R/'registration.json').exists():raise FileExistsError('No overwrite or automatic retry')
    z=np.load(RUN_ROOT/'private/cache/baseline.npz');y=z['y']
    if BASELINE:assert (RUN_ROOT/'results/active_record_control/verification.json').exists()
    if GAIN:assert (RUN_ROOT/'results/active_record_baseline/verification.json').exists()
    full=np.load(P/'features.npy') if GAIN else z['X'] if BASELINE else np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
    assert full.shape==(1920,19,WIDTH)
    split=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test']);meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
    schedule=schedules(dev,y[dev],meta.iloc[dev].recording.to_numpy())
    assert schedule==json.loads((RUN_ROOT/'private/record_reference_control/splits.json').read_text())
    (P/'splits.json').write_text(json.dumps(schedule))
    (R/'registration.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/('experiments/gain_invariance_v14/PLAN.md' if GAIN else 'experiments/active_record_control/BASELINE_PLAN.md' if BASELINE else 'experiments/active_record_control/PLAN.md')).read_bytes()).hexdigest(),
        new_fits=8,development_trials=1536,excluded_historical_test=384,primary_unit='200ms activity window',windows_per_trial=19,classes=8,rounds=400,trees=3200,features=WIDTH,
        candidate_selection=False,posthoc_baseline_control=BASELINE,posthoc_gain_control=GAIN,external_independent_data=False),indent=2)+'\n')
    supports=[]
    for s in schedule:
        tr=np.array(s['train']);te=np.array(s['evaluation']);stem=f"{s['protocol']}_{s['fold']}"
        assert not set(tr)&set(te) and set(tr)|set(te)==set(dev) and not(set(tr)|set(te))&held
        common=len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording));assert common==(64 if s['protocol']=='trial_shared' else 0)
        assert meta.iloc[tr].subject.nunique()==8
        for role,ids in [('train',tr),('evaluation',te)]:
            for c in range(8):
                m=meta.iloc[ids];m=m[m.label==c];assert len(m)>0
                supports.append(dict(protocol=s['protocol'],fold=s['fold'],role=role,label=c,trials=len(m),windows=len(m)*19,records=m.recording.nunique()))
        scaler=StandardScaler();a=scaler.fit_transform(full[tr].reshape(-1,WIDTH));b=scaler.transform(full[te].reshape(-1,WIDTH))
        model=make_model('XGB');model.set_params(n_jobs=6,callbacks=[TrainingProgress(NAME+'_'+stem,400)])
        start=perf_counter();model.fit(a,np.repeat(y[tr],19));elapsed=perf_counter()-start;model.set_params(callbacks=None)
        config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
        assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==3200
        np.testing.assert_array_equal(model.classes_,np.arange(8));q=model.get_booster().predict(DMatrix(b,nthread=6));pred=q.argmax(axis=1);truth=np.repeat(y[te],19)
        joblib.dump(dict(model=model,scaler=scaler),P/f'{stem}.joblib')
        np.savez(P/f'{stem}.npz',truth=truth,prediction=pred,probability=q,trial=np.repeat(te,19),scaler_mean=scaler.mean_,scaler_var=scaler.var_)
        (P/f'{stem}.json').write_text(json.dumps(config,indent=2))
        row=dict(protocol=s['protocol'],fold=s['fold'],train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),evaluation_windows=len(b),record_intersection=common,train_subjects=meta.iloc[tr].subject.nunique(),evaluation_subjects=meta.iloc[te].subject.nunique(),features=WIDTH,classes=8,rounds=400,trees=3200,fit_seconds=elapsed,model_bytes=(P/f'{stem}.joblib').stat().st_size,correct=int((truth==pred).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False)
        print(json.dumps(row),flush=True);del a,b,model,scaler;gc.collect()
    pd.DataFrame(supports).to_csv(R/'class_support.csv',index=False)
    (R/'complete.json').write_text(json.dumps(dict(new_fits=8,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
