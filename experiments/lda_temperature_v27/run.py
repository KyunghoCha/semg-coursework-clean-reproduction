"""Select a small LDA temperature only on existing inner observations."""
import json
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
import pandas as pd
from semg.evaluation import metrics
from sources27 import P,R,RUN_ROOT,SOURCE_ROOT,inputs,load,source_paths,snapshot,check_hashes,digest
from temperature import TEMPERATURES,soften,combine,choose


def nll(t,p):return float(-np.log(np.maximum(p[np.arange(len(t)),t],1e-15)).mean())


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Preserve previous execution; inspect before retry')
    beta=json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas'];(R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=digest(SOURCE_ROOT/'experiments/lda_temperature_v27/PLAN.md'),temperatures=TEMPERATURES,fold_betas=beta,new_fits=0,input_ms=200,windows_per_trial=19,historical_test_evaluations=0),indent=2)+'\n')
    x,y,target,schedule=inputs();(P/'splits.json').open('x').write(json.dumps(schedule));paths=[RUN_ROOT/'results/mixture_weights_v25/selection.json',RUN_ROOT/'private/splits.json',RUN_ROOT/'private/nested_selection_v10/splits.json']
    for stage in ['inner','outer']:
        for k in range(3):
            ps=source_paths(stage,k);paths.extend(ps);paths.extend(p.with_suffix('.joblib') for p in ps)
    hashes=snapshot(paths);(P/'source_hashes.json').open('x').write(json.dumps(hashes));rows=[];chosen=[];start=perf_counter()
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);parts=load('inner',k,ids,y);foldrows=[]
        for t in TEMPERATURES:
            q=combine(parts,beta[k],t);pred=q.argmax(1);row=dict(fold=k,temperature=t,windows=len(pred),correct=int((pred==truth).sum()),mixture_nll=nll(truth,q),lda_nll=nll(truth,soften(parts[1].reshape(-1,8,8).sum(1),t)),**metrics(truth,pred));foldrows.append(row);rows.append(row)
        chosen.append(choose(foldrows))
    pd.DataFrame(rows).to_csv(R/'inner.csv',index=False);selection=dict(fold_temperatures=chosen,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat());(R/'selection.json').open('x').write(json.dumps(selection,indent=2)+'\n');selection_sha=digest(R/'selection.json');selection_seconds=perf_counter()-start;print('FROZEN_SELECTION',selection,flush=True);rows=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['outer_evaluation']);truth=np.repeat(y[ids],19);parts=load('outer',k,ids,y)
        for role,t in [('incumbent',1.),('selected',chosen[k])]:
            q=combine(parts,beta[k],t);pred=q.argmax(1)
            if role=='incumbent':np.testing.assert_array_equal(q,np.load(RUN_ROOT/f'private/mixture_weights_v25/outer_selected_{k}.npz')['probability'])
            np.savez(P/f'outer_{role}_{k}.npz',trial=np.repeat(ids,19),truth=truth,probability=q,prediction=pred);rows.append(dict(role=role,fold=k,temperature=t,windows=len(pred),correct=int((pred==truth).sum()),mixture_nll=nll(truth,q),**metrics(truth,pred)))
        assert digest(R/'selection.json')==selection_sha
    pd.DataFrame(rows).to_csv(R/'outer.csv',index=False);check_hashes(hashes);(R/'complete.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),selection_sha256=selection_sha,selection_seconds=selection_seconds,new_fits=0,historical_test_evaluations=0),indent=2)+'\n');print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
