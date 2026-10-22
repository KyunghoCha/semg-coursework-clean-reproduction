"""Freeze small SVM weights from existing inner predictions, before any outer SVM fit."""
import hashlib,json,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import P as SVM_P,reference,to_eight
sys.path.insert(0,str(SOURCE_ROOT/'experiments/lda_blend_v19'))
from blend_rule import ALPHAS,mix,choose
P=RUN_ROOT/'private/svm_blend_v21';R=RUN_ROOT/'results/svm_blend_v21'


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    assert (RUN_ROOT/'results/svm410_v20/verification.json').exists()
    if (R/'registration.json').exists():raise FileExistsError('Preserve frozen selection')
    assert tuple(ALPHAS)==(0.,.05,.1,.2)
    plan=SOURCE_ROOT/'experiments/svm_blend_v21/PLAN.md'
    (R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),utc=datetime.now(timezone.utc).isoformat(),betas=ALPHAS,inner_new_fits=0,outer_calibrated_models_max=3,outer_base_svc_fits_max=12,features=410,learning_targets=64,output_classes=8,window_ms=200,windows_per_trial=19,development_trials=1536,historical_test_excluded=384,external_independent_test=False),indent=2)+'\n')
    rows=[];betas=[];source_hashes=[]
    for k in range(3):
        ref=reference('inner',k);path=SVM_P/f'inner_svm_{k}.npz';z=np.load(path)
        np.testing.assert_array_equal(ref['trial'],z['trial']);np.testing.assert_array_equal(ref['truth'],z['truth']);q=to_eight(z['probability']);source_hashes.append(dict(fold=k,svm_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));part=[]
        for b in ALPHAS:
            pred=mix(ref['probability'],q,b).argmax(1);part.append(dict(fold=k,alpha=b,correct=int((pred==z['truth']).sum()),windows=len(pred),**manual_metrics(z['truth'],pred)))
        rows.extend(part);betas.append(choose(part))
    totals=[dict(alpha=b,correct=sum(r['correct'] for r in rows if r['alpha']==b)) for b in ALPHAS]
    result=dict(fold_betas=betas,pooled_inner_beta=choose(totals),inner_correct_totals=totals,outer_scores_used=False,source_hashes=source_hashes)
    (R/'selection.json').open('x').write(json.dumps(result,indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'inner.csv',index=False);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
