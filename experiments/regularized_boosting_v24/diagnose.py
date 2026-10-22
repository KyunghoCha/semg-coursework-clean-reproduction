"""Summarize frozen v21 inner errors; never read outer predictions or fit a model."""
import json,sys
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import reference,to_eight
R=RUN_ROOT/'results/regularized_boosting_v24'


def main():
    if (R/'internal_diagnostic.json').exists():raise FileExistsError('Preserve diagnostic')
    splits=json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text());features=np.load(RUN_ROOT/'private/cache/baseline.npz')['X'];rms=np.sqrt(np.mean(features[:,:,:5]**2,axis=2));rows=[];positions=[];energy=[];confusions=[]
    for fold,s in enumerate(splits):
        ref=reference('inner',fold);sv=np.load(RUN_ROOT/f'private/svm410_v20/inner_svm_{fold}.npz');np.testing.assert_array_equal(ref['trial'],sv['trial']);np.testing.assert_array_equal(ref['truth'],sv['truth']);q=.8*ref['probability']+.2*to_eight(sv['probability']);truth=sv['truth'];pred=q.argmax(1);good=pred==truth;order=np.argsort(q,axis=1);top2=(order[:,-2:]==truth[:,None]).any(1);angle=pred//4!=truth//4;gesture=pred%4!=truth%4
        rows.append(dict(fold=fold,windows=len(truth),correct=int(good.sum()),errors=int((~good).sum()),true_class_in_top2_when_wrong=int((~good&top2).sum()),angle_only_errors=int((angle&~gesture).sum()),gesture_only_errors=int((~angle&gesture).sum()),both_errors=int((angle&gesture).sum())))
        cm=np.zeros((8,8),int);np.add.at(cm,(truth,pred),1);confusions.append(cm)
        for j in range(19):positions.append(dict(fold=fold,position=j,windows=len(truth)//19,correct=int(good.reshape(-1,19)[:,j].sum())))
        limits=np.quantile(rms[s['inner_train']].ravel(),[.25,.5,.75]);bins=np.searchsorted(limits,rms[s['inner_evaluation']].ravel())
        for b in range(4):energy.append(dict(fold=fold,training_rms_quartile=b,windows=int((bins==b).sum()),correct=int(good[bins==b].sum())))
    pd.DataFrame(rows).to_csv(R/'internal_errors.csv',index=False);pd.DataFrame(positions).to_csv(R/'internal_positions.csv',index=False);pd.DataFrame(energy).to_csv(R/'internal_energy.csv',index=False);cm=np.sum(confusions,axis=0);np.savetxt(R/'internal_confusion.csv',cm,fmt='%d',delimiter=',');pairs=sorted([(int(cm[a,b]),a,b) for a in range(8) for b in range(8) if a!=b],reverse=True)
    result=dict(scope='frozen v21 inner evaluation observations only',outer_predictions_read=False,historical_test_evaluated=False,new_fits=0,total={k:int(sum(row[k] for row in rows)) for k in rows[0] if k!='fold'},top_confusions=[dict(count=n,true_class=a,predicted_class=b) for n,a,b in pairs[:6]],internal_observations_are_not_full_oof=True,rms_thresholds_fit_on_inner_training_only=True)
    (R/'internal_diagnostic.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(pd.DataFrame(energy).groupby('training_rms_quartile')[['windows','correct']].sum().to_string())


if __name__=='__main__':main()
