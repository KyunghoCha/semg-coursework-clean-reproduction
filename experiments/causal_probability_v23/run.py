"""Select k from inner v21 probabilities, then apply it unchanged to outer trials."""
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.evaluation import metrics
from stream_average import HISTORIES,aggregate,choose
from context_inputs import P,R,load_source,snapshot,assert_preserved,schedule


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Inspect partial outputs before retry')
    assert (RUN_ROOT/'results/svm_blend_v21/verification.json').exists();plan=SOURCE_ROOT/'experiments/causal_probability_v23/PLAN.md';(R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),utc=datetime.now(timezone.utc).isoformat(),histories=HISTORIES,signal_ms=200,stride_ms=100,windows_per_trial=19,maximum_signal_spans_ms=[200,400,600],new_fits=0,development_trials=1536,historical_test_excluded=384,probability_source='frozen v21 full probabilities',v22_label_gate_used=False,known_trial_boundary_reset=True,future_windows_used=False,external_independent_test=False),indent=2)+'\n');(P/'splits.json').open('x').write(json.dumps(schedule()));hashes=snapshot([RUN_ROOT/'results/svm_blend_v21/selection.json',RUN_ROOT/'private/nested_selection_v10/splits.json',RUN_ROOT/'private/splits.json']);rows=[];choices=[]
    previous=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/inner.csv')
    for fold in range(3):
        q,truth,ids,paths=load_source('inner',fold);hashes.update(snapshot(paths));part=[]
        for k in HISTORIES:
            smooth=aggregate(q,k);pred=smooth.argmax(2);np.testing.assert_array_equal(smooth[:,0],q[:,0]);row=dict(fold=fold,history=k,windows=pred.size,maximum_signal_span_ms=200+100*(k-1),correct=int((pred==truth).sum()),first_window_label_changes=int((pred[:,0]!=q[:,0].argmax(1)).sum()),**metrics(truth.ravel(),pred.ravel()));part.append(row);np.savez(P/f'inner_k{k}_{fold}.npz',trial=np.repeat(ids,19),truth=truth.ravel(),probability=smooth.reshape(-1,8),prediction=pred.ravel())
            if k==1:assert row['correct']==int(previous[(previous.fold==fold)&(previous.alpha==.2)].iloc[0].correct)
        choices.append(choose(part));rows.extend(part)
    pd.DataFrame(rows).to_csv(R/'inner.csv',index=False);totals=[dict(history=k,correct=sum(r['correct'] for r in rows if r['history']==k)) for k in HISTORIES];selection=dict(fold_histories=choices,pooled_inner_history=choose(totals),inner_correct_totals=totals,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat());sel_path=R/'selection.json';sel_path.open('x').write(json.dumps(selection,indent=2)+'\n');sel_hash=hashlib.sha256(sel_path.read_bytes()).hexdigest();print('FROZEN_SELECTION',json.dumps(selection),flush=True);out=[]
    for fold,k in enumerate(choices):
        q,truth,ids,paths=load_source('outer',fold);hashes.update(snapshot(paths));smoothed=aggregate(q,k)
        for role,history,prob in [('current',1,q),('selected',k,smoothed)]:
            pred=prob.argmax(2);np.testing.assert_array_equal(pred[:,0],q[:,0].argmax(1));np.savez(P/f'outer_{role}_{fold}.npz',trial=np.repeat(ids,19),truth=truth.ravel(),probability=prob.reshape(-1,8),prediction=pred.ravel());row=dict(fold=fold,role=role,history=history,maximum_signal_span_ms=200+100*(history-1),windows=pred.size,correct=int((pred==truth).sum()),**metrics(truth.ravel(),pred.ravel()));out.append(row)
        print('OUTER',fold,out[-2:],flush=True);assert hashlib.sha256(sel_path.read_bytes()).hexdigest()==sel_hash
    assert_preserved(hashes);(P/'source_hashes.json').open('x').write(json.dumps(hashes,indent=2)+'\n');pd.DataFrame(out).to_csv(R/'outer.csv',index=False);(R/'complete.json').open('x').write(json.dumps(dict(new_fits=0,inner_candidates_evaluated=9,outer_protocol_comparisons=3,outer_windows_per_protocol=29184,historical_test_evaluations=0,selection_sha256=sel_hash,source_files_preserved=len(hashes),utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':main()
