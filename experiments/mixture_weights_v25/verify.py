"""Independent algebra and inner selection on verified fixed constituents."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
from sources import P,R,inputs,source_paths,digest,check_hashes


def probability(stage,fold,beta):
    arrays=[np.load(path)['probability'] for path in source_paths(stage,fold)]
    parts=[np.stack([q[:,c::8].sum(1) for c in range(8)],axis=1) for q in arrays]
    return (1-beta)*(.8*parts[0]+.2*parts[1])+beta*parts[2]


def main():
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    parent=json.loads((RUN_ROOT/'results/regularized_boosting_v24/verification.json').read_text());assert parent['models_restored']>=6
    complete=json.loads((R/'complete.json').read_text());reg=json.loads((R/'registration.json').read_text());assert digest(SOURCE_ROOT/'experiments/mixture_weights_v25/PLAN.md')==reg['plan_sha256'];assert digest(R/'selection.json')==complete['selection_sha256'];hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes)
    x,y,target,schedule=inputs();del x,target;old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);expected=[]
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+fold).split(tr,y[tr]));expected.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==expected==json.loads((P/'splits.json').read_text());inner=pd.read_csv(R/'inner.csv');choices=[]
    for fold,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);assert not set(ids)&set(s['inner_train']) and not set(ids)&held;rows=[]
        for beta in [0.,.1,.2,.3,.4,.5]:
            pred=probability('inner',fold,beta).argmax(1);row=inner[(inner.fold==fold)&(inner.beta==beta)].iloc[0];correct=int((pred==truth).sum());assert row.correct==correct
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            rows.append((correct,beta))
        choices.append(sorted(rows,key=lambda r:(-r[0],r[1]))[0][1])
    assert choices==json.loads((R/'selection.json').read_text())['fold_betas'];outer=pd.read_csv(R/'outer.csv');pool={};paired=[];positions=[];allids=[]
    for fold,(s,beta) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);allids.extend(ids);truth=np.repeat(y[ids],19);assert not set(ids)&set(s['outer_train']) and not set(ids)&held;preds={}
        for role,b in [('incumbent',.2),('selected',beta)]:
            q=probability('outer',fold,b);z=np.load(P/f'outer_{role}_{fold}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);row=outer[(outer.fold==fold)&(outer.role==role)].iloc[0];assert int((pred==truth).sum())==row.correct
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            pool.setdefault(role,[]).append((truth,pred));preds[role]=pred
            for j in range(19):positions.append(dict(fold=fold,role=role,position=j,windows=len(ids),correct=int((pred==truth).reshape(-1,19)[:,j].sum())))
        a=preds['incumbent']==truth;b=preds['selected']==truth;ta=a.reshape(-1,19).sum(1);tb=b.reshape(-1,19).sum(1);paired.append(dict(fold=fold,corrected=int((b&~a).sum()),broken=int((a&~b).sum()),net=int(b.sum()-a.sum()),trial_improved=int((tb>ta).sum()),trial_worsened=int((tb<ta).sum()),trial_tied=int((tb==ta).sum())))
    assert sorted(allids)==sorted(dev);pooled=[]
    for role,parts in pool.items():
        t=np.concatenate([a for a,b in parts]);p=np.concatenate([b for a,b in parts]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(t,p)));cm=np.zeros((8,8),int);np.add.at(cm,(t,p),1);pd.DataFrame(cm).to_csv(R/f'confusion_{role}.csv',index=False)
    prior=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/pooled.csv').query("role=='selected'").iloc[0];assert pooled[0]['correct']==prior.correct
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False);check_hashes(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(inner_candidates_recomputed=18,outer_probability_rows=58368,inner_only_choice_verified=True,splits_reconstructed=True,source_hashes_preserved=True,all_1536_development_trials_and_19_windows=True,historical_test_intersection=0,temporal_context=False,input_ms=200,new_model_fits=0,prior_checkpoint_verification='results/regularized_boosting_v24/verification.json and results/svm_blend_v21/verification.json',external_independent_test=False),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':main()
