"""Select only small inner LDA weights; fit outer LDA only where selected."""
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
from common import P,R,inputs,baseline,make_lda,to_eight
from blend_rule import ALPHAS,mix,choose


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Review partial artifacts before retry')
    x,y,target,schedule=inputs();(P/'splits.json').open('x').write(json.dumps(schedule))
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test'])
    (R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/lda_blend_v19/PLAN.md').read_bytes()).hexdigest(),alphas=ALPHAS,inner_new_fits=0,outer_fits_max=3,development_trials=1536,excluded_historical_test=384,features=410,external_independent_test=False),indent=2)+'\n')
    choices=[];inner=[]
    for k,s in enumerate(schedule):
        z=np.load(baseline('inner',k));l=np.load(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{k}.npz');np.testing.assert_array_equal(z['trial'],l['trial']);np.testing.assert_array_equal(z['truth'],l['truth']);q=to_eight(z['probability']);ql=to_eight(l['probability'])
        np.testing.assert_array_equal(q.argmax(1),z['prediction']);rows=[]
        for alpha in ALPHAS:
            pred=mix(q,ql,alpha).argmax(1);rows.append(dict(fold=k,alpha=alpha,correct=int((pred==z['truth']).sum()),windows=len(pred),**metrics(z['truth'],pred)))
        inner.extend(rows);choices.append(choose(rows))
    totals=[dict(alpha=a,correct=sum(r['correct'] for r in inner if r['alpha']==a)) for a in ALPHAS]
    selection=dict(fold_alphas=choices,final_development_alpha=choose(totals),inner_correct_totals=totals,outer_scores_used=False)
    (R/'selection.json').open('x').write(json.dumps(selection,indent=2)+'\n');pd.DataFrame(inner).to_csv(R/'inner.csv',index=False);print('FROZEN_SELECTION',selection,flush=True)
    fits=[];out=[]
    for k,(alpha,s) in enumerate(zip(choices,schedule)):
        tr=np.array(s['outer_train']);te=np.array(s['outer_evaluation']);assert not set(tr)&set(te) and set(tr)|set(te)==dev and not(set(tr)|set(te))&held
        z=np.load(baseline('outer',k));np.testing.assert_array_equal(z['trial'],np.repeat(te,19));q=to_eight(z['probability']);ql=np.zeros_like(q)
        if alpha>0:
            labels=np.repeat(target[tr],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64));sc=StandardScaler();a=sc.fit_transform(x[tr].reshape(-1,410));b=sc.transform(x[te].reshape(-1,410));m=make_lda();start=perf_counter();m.fit(a,labels);seconds=perf_counter()-start;prob=m.predict_proba(b);ql=to_eight(prob)
            path=P/f'outer_lda_{k}.joblib';joblib.dump(dict(model=m,scaler=sc),path);np.savez(P/f'outer_lda_{k}.npz',train=tr,trial=np.repeat(te,19),truth=np.repeat(y[te],19),probability=prob,prediction=ql.argmax(1))
            fits.append(dict(fold=k,fit_seconds=seconds,model_bytes=path.stat().st_size,train_trials=len(tr),evaluation_trials=len(te),utc=datetime.now(timezone.utc).isoformat()));print('NEW_FIT',fits[-1],flush=True)
        joined=q if alpha==0 else mix(q,ql,alpha);truth=np.repeat(y[te],19);pred=joined.argmax(1)
        np.savez(P/f'outer_selected_{k}.npz',truth=truth,trial=np.repeat(te,19),probability=joined,prediction=pred)
        for role,p in [('baseline',q.argmax(1)),('selected',pred)]:out.append(dict(fold=k,role=role,alpha=0 if role=='baseline' else alpha,correct=int((truth==p).sum()),**metrics(truth,p)))
    pd.DataFrame(fits,columns=['fold','fit_seconds','model_bytes','train_trials','evaluation_trials','utc']).to_csv(R/'new_fits.csv',index=False);pd.DataFrame(out).to_csv(R/'outer.csv',index=False)
    (R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(fits),inner_new_fits=0,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
