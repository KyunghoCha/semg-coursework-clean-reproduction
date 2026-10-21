"""Fixed-factor comparisons on the existing development folds, no holdout access."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from semg.evaluation import folds,unfold,metrics
from semg.models import make_model,predict
ROOT=RUN_ROOT;P=ROOT/'private/mechanisms_v3';R=ROOT/'results/mechanisms_v3'
NAMES=['ZC_added_200_19','base_200_18','spectrum_200_18','spectrum_300_18','spectrum_200_19','spectrum_no60_200_19']


if __name__=='__main__':
    if (R/'scores.csv').exists():raise FileExistsError('Preserve the existing experiment')
    split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test'])
    y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y'];features=np.load(P/'features.npz',allow_pickle=False)
    schedule=[dict(train=dev[a].tolist(),validation=dev[b].tolist()) for a,b in folds(y[dev])]
    (P/'splits.json').write_text(json.dumps(schedule,indent=2))
    (R/'registration.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/mechanisms_v3/PLAN.md').read_bytes()).hexdigest(),
        candidates=NAMES,development_trials=1536,holdout_excluded=384,folds=5,selection=False,existing_development_reuse=True),indent=2))
    for name in NAMES:
        x=features[name];n=x.shape[1]
        for k,s in enumerate(schedule):
            tr=np.array(s['train']);te=np.array(s['validation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
            a,ya,ga=unfold(x,y,tr);b,yb,gb=unfold(x,y,te)
            scaler=StandardScaler();t=perf_counter();a=scaler.fit_transform(a);model=make_model('XGB');model.set_params(n_jobs=6);model.fit(a,ya)
            config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0';assert model.get_booster().num_boosted_rounds()==400
            elapsed=perf_counter()-t;b=scaler.transform(b);pred=predict(model,b)
            row=dict(candidate=name,fold=k,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),evaluation_windows=len(b),
                features=x.shape[-1],windows_per_trial=n,fit_seconds=elapsed,**metrics(yb,pred))
            stem=f'{name}_{k}';np.savez(P/f'{stem}.npz',truth=yb,prediction=pred,trial=gb,scaler_mean=scaler.mean_,scaler_var=scaler.var_)
            (P/f'{stem}.json').write_text(json.dumps(config,indent=2))
            if k==0:joblib.dump(dict(scaler=scaler,model=model),P/f'{name}.joblib')
            pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not (R/'scores.csv').exists(),index=False)
            print(json.dumps(row),flush=True);del model,a,b;gc.collect()
    (R/'complete.json').write_text(json.dumps(dict(fits=30,holdout_evaluations=0),indent=2))
