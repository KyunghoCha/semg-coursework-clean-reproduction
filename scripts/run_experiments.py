"""Whole-dataset baseline; development-only candidate selection; fixed holdout evaluation with documented prior CV use."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_rel
import sklearn
from sklearn.preprocessing import StandardScaler
import xgboost

from semg.evaluation import check_split, unfold, metrics, majority, folds
from semg.features import feature_dataset
from semg.models import make_model, predict
from semg.paths import RUN_ROOT, SOURCE_ROOT

ROOT = RUN_ROOT
PRIVATE = ROOT/"private"
RESULTS = ROOT/"results"
BASE = ["SVM", "RF", "XGB"]
CANDIDATES = BASE + ["SVM_C10", "XGB_depth4", "XGB_spectral", "XGB_win300", "XGB_forearm"]


def fit_evaluate(name, features, y, tr, te, phase, fold):
    check_split(tr, te, len(y))
    a, ya, ga = unfold(features, y, tr)
    b, yb, gb = unfold(features, y, te)
    assert not set(ga) & set(gb)
    scaler = StandardScaler()
    model = make_model(name)
    t = perf_counter()
    a = scaler.fit_transform(a)
    expected_mean = features[tr].reshape(-1, features.shape[-1]).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-12, atol=1e-12)
    model.fit(a, ya)
    train_seconds = perf_counter()-t
    b = scaler.transform(b)
    if name.startswith("XGB"):
        config = json.loads(model.get_booster().save_config())
        assert config["learner"]["generic_param"]["device"].startswith("cuda")
        assert model.get_booster().num_boosted_rounds() == 400
    else:
        config = model.get_params()
        if name == "RF": assert len(model.estimators_) == 100
    p = predict(model, b)
    ids, vote = majority(p, gb)
    row = dict(model=name, phase=phase, fold=fold, train_trials=len(tr),
               test_trials=len(te), train_windows=len(a), test_windows=len(b),
               features=features.shape[-1], shared_trials=0, train_seconds=train_seconds,
               **metrics(yb, p), trial_vote_accuracy=float(np.mean(y[ids] == vote)))
    folder = PRIVATE/phase
    folder.mkdir(exist_ok=True)
    np.savez(folder/f"{name}_{fold}_predictions.npz", truth=yb, prediction=p,
             trial=gb, trial_vote=vote, trial_ids=ids)
    (folder/f"{name}_{fold}_parameters.json").write_text(json.dumps(
        dict(model=config, scaler_mean=scaler.mean_.tolist(), scaler_var=scaler.var_.tolist(),
             scaler_n=int(scaler.n_samples_seen_)), indent=2))
    if phase in ("baseline_holdout", "selected_holdout"):
        predict(model, b[:1])
        t = perf_counter()
        for _ in range(100): predict(model, b[:1])
        row['inference_ms_per_window'] = (perf_counter()-t)*10
        dest = folder/f"{name}.joblib"
        joblib.dump(dict(scaler=scaler, model=model), dest, compress=0)
        row['model_megabytes'] = dest.stat().st_size/1e6
    print(json.dumps(row), flush=True)
    return row


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def load():
    d = np.load(PRIVATE/"cache/baseline.npz", allow_pickle=False)
    split = json.loads((PRIVATE/"splits.json").read_text())
    return d['X'], d['y'], split


def baseline():
    target = RESULTS/"baseline.csv"
    if target.exists(): raise FileExistsError("Baseline already exists; use a new run directory")
    X, y, split = load()
    for phase, schedule in [("baseline_holdout", [split]), ("baseline_cv", split['folds'])]:
        for k,s in enumerate(schedule):
            for name in BASE:
                append_row(target, fit_evaluate(name, X, y, np.array(s['train']),
                           np.array(s['test']), phase, k))
    df = pd.read_csv(target)
    cv = df[df.phase == 'baseline_cv']
    rows = []
    for a,b in [("SVM", "RF"), ("SVM", "XGB"), ("RF", "XGB")]:
        xa=cv[cv.model == a].sort_values('fold').accuracy.to_numpy()
        xb=cv[cv.model == b].sort_values('fold').accuracy.to_numpy()
        t,p = ttest_rel(xa,xb)
        rows.append(dict(model_a=a, model_b=b, t=t, p=p, n=5,
                         interpretation="exploratory_dependent_folds"))
    pd.DataFrame(rows).to_csv(RESULTS/"paired_tests.csv", index=False)
    (RESULTS/"baseline_complete.json").write_text(json.dumps(
        dict(fits=len(df), models=BASE, all_folds_complete=len(df)==18,
             protocol_sha256=hashlib.sha256((SOURCE_ROOT/'docs/PROTOCOL.md').read_bytes()).hexdigest()),indent=2))


def variant_features(name, baseline_X):
    if name not in ("XGB_spectral", "XGB_win300", "XGB_forearm"):
        return baseline_X
    cache = PRIVATE/f"cache/{name}.npy"
    if cache.exists(): return np.load(cache, allow_pickle=False)
    seg = np.load(PRIVATE/'cache/segments.npy', allow_pickle=False)
    f = feature_dataset(seg, length=300 if name == "XGB_win300" else 200,
                        spectral=name == "XGB_spectral",
                        channels=(0,1,2) if name == "XGB_forearm" else (0,1,2,3,4))
    np.save(cache, f)
    return f


def improve():
    if not (RESULTS/"baseline_complete.json").exists():
        raise RuntimeError("Complete the full baseline first")
    target = RESULTS/"development.csv"
    if target.exists(): raise FileExistsError("Development run already exists")
    X, y, split = load()
    outer_tr=np.array(split['train']);dev_y=y[outer_tr]
    schedule=folds(dev_y)
    for name in CANDIDATES:
        f = variant_features(name, X)[outer_tr]
        for k,(a,b) in enumerate(schedule):
            append_row(target, fit_evaluate(name, f, dev_y, a,b,"development", k))
    df=pd.read_csv(target)
    means=df.groupby('model').accuracy.mean()
    eligible=CANDIDATES[:-1]
    chosen=max(eligible,key=lambda n: means[n])
    selection=dict(model=chosen, criterion="development_mean_window_accuracy",
                   eligible=eligible, means=means.to_dict(), test_consulted_for_selection=False,
                   selection_scope="development scores only; prior baseline CV used all holdout trials")
    (RESULTS/'selection.json').write_text(json.dumps(selection,indent=2))
    print('SELECTION',json.dumps(selection),flush=True)
    if chosen in BASE:
        print('Selected baseline: existing holdout evaluation retained.',flush=True)
        return
    f=variant_features(chosen,X)
    row=fit_evaluate(chosen,f,y,outer_tr,np.array(split['test']),'selected_holdout',0)
    pd.DataFrame([row]).to_csv(RESULTS/'selected_holdout.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','improve']);args=p.parse_args()
    env=dict(python=sys.version,platform=platform.platform(),
             numpy=np.__version__,pandas=pd.__version__,
             scipy=scipy.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,
             xgboost_build=xgboost.build_info())
    (PRIVATE/f'environment_{args.phase}.json').write_text(json.dumps(env,indent=2))
    if args.phase=='baseline':baseline()
    else:improve()


if __name__=='__main__':main()
