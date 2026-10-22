"""Rebuild trial chronology, scalar past-only means, selection, and error distributions."""
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
from context_inputs import P,R,baseline,assert_preserved
from stream_average import aggregate

PRECISION=[]


def scalar_means(q,k):
    out=np.empty_like(q,dtype=float)
    for i in range(len(q)):
        for j in range(19):
            left=max(0,j-k+1)
            for c in range(8):out[i,j,c]=sum(float(q[i,h,c]) for h in range(left,j+1))/(j-left+1)
    return out


def source(stage,fold,ids,y):
    if stage=='inner':
        arrays=[np.load(baseline('inner',fold)),np.load(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{fold}.npz'),np.load(RUN_ROOT/f'private/svm410_v20/inner_svm_{fold}.npz')]
        for z in arrays:
            np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[ids],19))
        parts=[z['probability'].reshape(-1,8,8).sum(1) for z in arrays];q=.8*(.8*parts[0]+.2*parts[1])+.2*parts[2]
        high=[z['probability'].astype(float).reshape(-1,8,8).sum(1) for z in arrays];high=.64*high[0]+.16*high[1]+.2*high[2]
        differences=[]
        for history in [1,3,5]:
            a=scalar_means(q.reshape(-1,19,8),history).argmax(2);b=scalar_means(high.reshape(-1,19,8),history).argmax(2);differences.append(int((a!=b).sum()));np.testing.assert_array_equal(a,b)
        PRECISION.append(dict(fold=fold,max_probability_difference=float(np.max(np.abs(q-high))),label_changes_by_history=differences))
    else:
        z=np.load(RUN_ROOT/f'private/svm_blend_v21/outer_selected_{fold}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[ids],19));q=z['probability'].astype(float)
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6);return q.reshape(-1,19,8)


def run_counts(wrong):
    counts=np.zeros(20,int)
    for row in wrong:
        length=0
        for flag in list(row)+[False]:
            if flag:length+=1
            elif length:counts[length]+=1;length=0
    assert int(np.arange(20)@counts)==int(wrong.sum());return counts


def main():
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    complete=json.loads((R/'complete.json').read_text());reg=json.loads((R/'registration.json').read_text());assert hashlib.sha256((SOURCE_ROOT/'experiments/causal_probability_v23/PLAN.md').read_bytes()).hexdigest()==reg['plan_sha256'];sel_path=R/'selection.json';assert hashlib.sha256(sel_path.read_bytes()).hexdigest()==complete['selection_sha256'];sel=json.loads(sel_path.read_text());hashes=json.loads((P/'source_hashes.json').read_text());assert_preserved(hashes);old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];schedule=[]
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+fold).split(tr,y[tr]));schedule.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==json.loads((P/'splits.json').read_text());inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');totals={k:0 for k in [1,3,5]};choices=[];checked=0
    for fold,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);assert not set(ids)&held and not set(ids)&set(s['inner_train']);truth=np.repeat(y[ids],19);q=source('inner',fold,ids,y);scores=[]
        for k in [1,3,5]:
            expected=scalar_means(q,k);z=np.load(P/f'inner_k{k}_{fold}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(z['probability'],expected.reshape(-1,8),atol=1e-12,rtol=1e-12);pred=expected.argmax(2).ravel();np.testing.assert_array_equal(pred,z['prediction']);row=inner[(inner.fold==fold)&(inner.history==k)].iloc[0];correct=int((pred==truth).sum());assert row.correct==correct and row.first_window_label_changes==0;totals[k]+=correct;scores.append((correct,k));checked+=len(pred)
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            changed=q.copy();changed[:,9:]=changed[:,9:,::-1];np.testing.assert_array_equal(aggregate(q,k)[:,:9],aggregate(changed,k)[:,:9])
        choices.append(min(scores,key=lambda v:(-v[0],v[1]))[1])
    assert choices==sel['fold_histories'] and min(totals,key=lambda k:(-totals[k],k))==sel['pooled_inner_history'];assert sel['inner_correct_totals']==[dict(history=k,correct=totals[k]) for k in [1,3,5]]
    pool={};paired=[];positions=[];axes=[];runs=[];matrices={r:np.zeros((8,8),int) for r in ['current','selected']};ids_all=[]
    for fold,(k,s) in enumerate(zip(choices,schedule)):
        ids=np.array(s['outer_evaluation']);assert not set(ids)&held and not set(ids)&set(s['outer_train']);ids_all.extend(ids);truth=np.repeat(y[ids],19);q=source('outer',fold,ids,y);preds={}
        for role,history in [('current',1),('selected',k)]:
            expected=scalar_means(q,history);z=np.load(P/f'outer_{role}_{fold}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(z['probability'],expected.reshape(-1,8),atol=1e-12,rtol=1e-12);pred=expected.argmax(2).ravel();np.testing.assert_array_equal(pred,z['prediction']);row=outer[(outer.fold==fold)&(outer.role==role)].iloc[0];assert row.correct==int((pred==truth).sum());checked+=len(pred)
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            pool.setdefault(role,[]).append((truth,pred));preds[role]=pred.reshape(-1,19);good=(pred==truth).reshape(-1,19)
            for j in range(19):positions.append(dict(fold=fold,role=role,history=history,position=j,start_ms=1500+100*j,actual_signal_span_ms=200+100*min(j,history-1),windows=len(ids),correct=int(good[:,j].sum())))
            angle=pred//4!=truth//4;gesture=pred%4!=truth%4;axes.append(dict(fold=fold,role=role,angle_only=int((angle&~gesture).sum()),gesture_only=int((~angle&gesture).sum()),both=int((angle&gesture).sum())));np.add.at(matrices[role],(truth,pred),1);counts=run_counts(~good)
            for length in range(1,20):runs.append(dict(fold=fold,role=role,run_length=length,runs=int(counts[length])))
        current=preds['current']==truth.reshape(-1,19);selected=preds['selected']==truth.reshape(-1,19);np.testing.assert_array_equal(preds['current'][:,0],preds['selected'][:,0]);a=current.sum(1);b=selected.sum(1);paired.append(dict(fold=fold,corrected=int((selected&~current).sum()),broken=int((current&~selected).sum()),net=int(selected.sum()-current.sum()),first_window_label_changes=0,trial_improved=int((b>a).sum()),trial_worsened=int((b<a).sum()),trial_tied=int((b==a).sum())))
    assert sorted(ids_all)==sorted(dev);pooled=[]
    for role,parts in pool.items():
        truth=np.concatenate([a for a,b in parts]);pred=np.concatenate([b for a,b in parts]);pooled.append(dict(role=role,windows=len(truth),correct=int((truth==pred).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(truth,pred)));pd.DataFrame(matrices[role],index=[f'true_{i}' for i in range(8)],columns=[f'pred_{i}' for i in range(8)]).to_csv(R/f'confusion_{role}.csv')
    prior=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/pooled.csv').query("role=='selected'").iloc[0]
    for key,value in manual_metrics(np.concatenate([a for a,b in pool['current']]),np.concatenate([b for a,b in pool['current']])).items():assert abs(value-prior[key])<1e-12
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False);pd.DataFrame(axes).to_csv(R/'error_axes.csv',index=False);pd.DataFrame(runs).to_csv(R/'error_runs.csv',index=False);assert_preserved(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(scalar_probability_rows_verified=checked,frozen_plan_and_selection_preserved=True,trial_splits_and_19_row_chronology_verified=True,inner_choice_recomputed=True,prefix_invariance_to_future_changes=True,known_trial_boundaries_reset=True,first_window_label_changes=0,k1_matches_v21=True,all_1536_development_trials_evaluated_once=True,historical_test_intersection=0,new_model_fits=0,source_files_preserved=len(hashes),float64_algebraic_sensitivity=PRECISION,original_mixed_precision_arithmetic_reproduced=True,single_200ms_result_replaced=False,external_independent_test=False),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False));print(pd.DataFrame(paired).to_string(index=False))


if __name__=='__main__':main()
