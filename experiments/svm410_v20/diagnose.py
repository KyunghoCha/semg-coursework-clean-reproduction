"""Internal-only errors of the already frozen v19 mixture; no outer reads or fitting."""
import json,sys
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.append(str(SOURCE_ROOT/'experiments/shared_lda_v18'))
from lda_common import baseline,to_eight
R=RUN_ROOT/'results/svm410_v20'


def main():
    if (R/'internal_diagnostic.json').exists():raise FileExistsError('Preserve diagnostic')
    schedule=json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text());alpha=json.loads((RUN_ROOT/'results/lda_blend_v19/selection.json').read_text())['fold_alphas'];cache=np.load(RUN_ROOT/'private/cache/baseline.npz');rms=np.sqrt(np.mean(cache['X'][:,:,:5]**2,axis=2))
    positions=[];energy=[];cms=[];axes=[]
    for k,s in enumerate(schedule):
        xb=np.load(baseline('inner',k));ld=np.load(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{k}.npz');np.testing.assert_array_equal(xb['trial'],ld['trial']);truth=xb['truth'];pred=((1-alpha[k])*to_eight(xb['probability'])+alpha[k]*to_eight(ld['probability'])).argmax(1)
        cm=np.zeros((8,8),int);np.add.at(cm,(truth,pred),1);cms.append(cm)
        aa=truth//4==pred//4;gg=truth%4==pred%4;axes.append(dict(fold=k,windows=len(truth),correct=int((truth==pred).sum()),angle_only_errors=int((~aa&gg).sum()),gesture_only_errors=int((aa&~gg).sum()),both_errors=int((~aa&~gg).sum())))
        correct=(truth==pred).reshape(-1,19)
        positions.extend(dict(fold=k,start_ms=1500+j*100,windows=len(correct),correct=int(correct[:,j].sum()),accuracy=float(correct[:,j].mean())) for j in range(19))
        thresholds=np.quantile(rms[s['inner_train']].ravel(),[.25,.5,.75]);bins=np.searchsorted(thresholds,rms[s['inner_evaluation']].ravel());acc=(truth==pred)
        energy.extend(dict(fold=k,train_rms_quartile=j,windows=int((bins==j).sum()),correct=int(acc[bins==j].sum()),accuracy=float(acc[bins==j].mean())) for j in range(4))
    pd.DataFrame(positions).to_csv(R/'internal_positions.csv',index=False);pd.DataFrame(energy).to_csv(R/'internal_energy.csv',index=False);pd.DataFrame(axes).to_csv(R/'internal_error_axes.csv',index=False);total=np.sum(cms,axis=0);np.savetxt(R/'internal_confusion.csv',total,fmt='%d',delimiter=',')
    err=total.copy();np.fill_diagonal(err,0);pairs=sorted([(int(err[a,b]),a,b) for a in range(8) for b in range(8) if a!=b],reverse=True)[:6]
    summary=dict(scope='v19 fixed mixture inner evaluation observations only',outer_predictions_read=False,historical_test_evaluated=False,observations=int(total.sum()),correct=int(np.trace(total)),errors=int(err.sum()),top_confusions=[dict(count=n,true_class=a,predicted_class=b) for n,a,b in pairs],axes_totals={key:sum(r[key] for r in axes) for key in ['angle_only_errors','gesture_only_errors','both_errors']},rms_bins_fit_on_inner_training_only=True,internal_observations_are_not_full_oof=True)
    (R/'internal_diagnostic.json').open('x').write(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(pd.DataFrame(positions).groupby('start_ms')[['windows','correct']].sum().assign(accuracy=lambda d:d.correct/d.windows).to_string());print(pd.DataFrame(energy).groupby('train_rms_quartile')[['windows','correct']].sum().assign(accuracy=lambda d:d.correct/d.windows).to_string())


if __name__=='__main__':main()
