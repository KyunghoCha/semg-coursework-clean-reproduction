"""Freeze beta on inner observations, then score its unchanged outer mixture."""
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.evaluation import metrics
from sources import P,R,inputs,load,source_paths,snapshot,check_hashes,digest
from weights import BETAS,combine,choose


def main():
    assert (RUN_ROOT/'results/regularized_boosting_v24/verification.json').exists()
    if (R/'registration.json').exists():raise FileExistsError('Inspect prior outputs before retry')
    x,y,target,schedule=inputs();del x,target
    (P/'splits.json').open('x').write(json.dumps(schedule));plan=SOURCE_ROOT/'experiments/mixture_weights_v25/PLAN.md'
    (R/'registration.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(plan),betas=BETAS,signal_ms=200,windows_per_trial=19,new_fits=0,temporal_context=False),indent=2)+'\n')
    hashes=snapshot([RUN_ROOT/'results/regularized_boosting_v24/selection.json',RUN_ROOT/'private/nested_selection_v10/splits.json',RUN_ROOT/'private/splits.json']);inner=[];betas=[]
    for fold,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);qs=load('inner',fold,ids,y);hashes.update(snapshot(source_paths('inner',fold)));truth=np.repeat(y[ids],19);rows=[]
        for beta in BETAS:
            q=combine(*qs,beta);pred=q.argmax(1);rows.append(dict(fold=fold,beta=beta,windows=len(pred),correct=int((pred==truth).sum()),**metrics(truth,pred)))
        betas.append(choose(rows));inner.extend(rows)
    pd.DataFrame(inner).to_csv(R/'inner.csv',index=False);selection=dict(fold_betas=betas,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat());(R/'selection.json').open('x').write(json.dumps(selection,indent=2)+'\n');sha=digest(R/'selection.json');print('FROZEN_SELECTION',json.dumps(selection),sha,flush=True);out=[]
    for fold,(s,beta) in enumerate(zip(schedule,betas)):
        ids=np.array(s['outer_evaluation']);qs=load('outer',fold,ids,y);hashes.update(snapshot(source_paths('outer',fold)));truth=np.repeat(y[ids],19)
        for role,b in [('incumbent',.2),('selected',beta)]:
            q=combine(*qs,b);pred=q.argmax(1)
            if role=='incumbent':np.testing.assert_array_equal(pred,np.load(RUN_ROOT/f'private/regularized_boosting_v24/comparison_selected_{fold}.npz')['prediction'])
            np.savez(P/f'outer_{role}_{fold}.npz',trial=np.repeat(ids,19),truth=truth,probability=q,prediction=pred);out.append(dict(fold=fold,role=role,beta=b,windows=len(pred),correct=int((pred==truth).sum()),**metrics(truth,pred)))
        assert digest(R/'selection.json')==sha
    pd.DataFrame(out).to_csv(R/'outer.csv',index=False);check_hashes(hashes);(P/'source_hashes.json').open('x').write(json.dumps(hashes,indent=2)+'\n');(R/'complete.json').open('x').write(json.dumps(dict(new_fits=0,selection_sha256=sha,utc=datetime.now(timezone.utc).isoformat(),historical_test_evaluations=0),indent=2)+'\n');print(pd.DataFrame(out).to_string(index=False),flush=True)


if __name__=='__main__':main()
