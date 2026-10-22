"""One frozen covariance classifier, with inner-only choice and existing baseline reuse."""
import hashlib,json
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.evaluation import metrics
from gaussian_boundary import ShrinkQDA,to_eight
from qda_data import P,R,inputs,baseline


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Review partial artifacts before any retry')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test'])
    (P/'splits.json').open('x').write(json.dumps(schedule))
    (R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/shrink_qda_v17/PLAN.md').read_bytes()).hexdigest(),development_trials=1536,excluded_historical_test=384,features=410,learning_targets=64,window_ms=200,windows_per_trial=19,new_fits_range=[3,6],cpu_threads=6,selection='inner correct counts, strict improvement over narrow64',external_independent_test=False),indent=2)+'\n')
    rows=[]
    def fit(stage,k,tr,te):
        tr=np.array(tr);te=np.array(te);stem=f'{stage}_qda_{k}'
        assert not set(tr)&set(te) and (set(tr)|set(te))<=dev and not(set(tr)|set(te))&held
        if stage=='outer':assert set(tr)|set(te)==dev
        labels=np.repeat(target[tr],19);counts=np.bincount(labels,minlength=64);assert np.array_equal(np.unique(labels),np.arange(64))
        scaler=StandardScaler();a=scaler.fit_transform(x[tr].reshape(-1,410));b=scaler.transform(x[te].reshape(-1,410))
        print('START',stem,datetime.now(timezone.utc).isoformat(),flush=True)
        start=perf_counter();m=ShrinkQDA().fit(a,labels);seconds=perf_counter()-start
        q=m.predict_proba(b);assert q.shape==(len(te)*19,64);np.testing.assert_allclose(q.sum(1),1,atol=1e-9)
        pred=to_eight(q).argmax(1);truth=np.repeat(y[te],19);joblib.dump(dict(model=m,scaler=scaler),P/f'{stem}.joblib')
        np.savez(P/f'{stem}.npz',truth=truth,prediction=pred,trial=np.repeat(te,19),probability=q,train=tr)
        np.savez(P/f'{stem}_diagnostics.npz',class_support=counts,shrinkage=m.shrinkage_)
        row=dict(stage=stage,fold=k,model='qda',utc=datetime.now(timezone.utc).isoformat(),features=410,learning_classes=64,train_trials=len(tr),evaluation_trials=len(te),evaluation_windows=len(truth),minimum_class_train_windows=int(counts.min()),maximum_class_train_windows=int(counts.max()),fit_seconds=seconds,model_bytes=(P/f'{stem}.joblib').stat().st_size,correct=int((truth==pred).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False);rows.append(row);print('DONE',json.dumps(row),flush=True);return row
    inner=[fit('inner',k,s['inner_train'],s['inner_evaluation']) for k,s in enumerate(schedule)]
    references=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/inner.csv');choices=[];comparison=[]
    for k,row in enumerate(inner):
        ref=references[(references.model=='narrow64')&(references.fold==k)].iloc[0];chosen='qda' if row['correct']>int(ref.correct) else 'narrow64';choices.append(chosen)
        z=np.load(baseline('inner',k));np.testing.assert_array_equal(z['truth'],np.repeat(y[schedule[k]['inner_evaluation']],19));assert int((z['truth']==z['prediction']).sum())==int(ref.correct)
        comparison.append(dict(fold=k,qda_correct=row['correct'],baseline_correct=int(ref.correct),selected=chosen))
    (R/'selection.json').open('x').write(json.dumps(dict(choices=choices,comparisons=comparison,outer_scores_used=False),indent=2)+'\n');print('FROZEN_SELECTION',choices,flush=True)
    for k,(name,s) in enumerate(zip(choices,schedule)):
        if name=='qda':fit('outer',k,s['outer_train'],s['outer_evaluation'])
    (R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(rows),inner_fits=3,outer_fits=len(rows)-3,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
