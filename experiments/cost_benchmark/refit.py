"""Rebuild four unsaved inner models strictly for cost measurement, verify old predictions."""
from semg.paths import RUN_ROOT,SOURCE_ROOT
import importlib.util
import json
import sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from semg.models import predict
spec=importlib.util.spec_from_file_location('capacity_v4_run',SOURCE_ROOT/'experiments/capacity_v4/run.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
P=RUN_ROOT/'private';R=RUN_ROOT/'results/cost_benchmark'
schedule=json.loads((P/'capacity_v4/splits.json').read_text())[0]['inner'][0]
tr=np.array(schedule['train']);te=np.array(schedule['validation']);held=set(json.loads((P/'splits.json').read_text())['test'])
assert not (set(tr)|set(te))&held and not set(tr)&set(te)
rich=np.load(P/'diagnostic_v2/features.npz',allow_pickle=False)['rich'];lean=np.concatenate([rich[:,:,:25],rich[:,:,110:]],axis=2)
y=np.load(P/'cache/baseline.npz',allow_pickle=False)['y'];rows=[]
for name in ['XGB_rich800','XGB_noTime800','SVM_rich_standard','SVM_rich_quantile']:
    dest=P/f'cost_benchmark/refit_{name}.joblib'
    if dest.exists():raise FileExistsError(dest)
    x=lean if name=='XGB_noTime800' else rich;a=x[tr].reshape(-1,x.shape[-1]);b=x[te].reshape(-1,x.shape[-1])
    scaler=QuantileTransformer(n_quantiles=1000,output_distribution='normal',subsample=None,random_state=42) if name=='SVM_rich_quantile' else StandardScaler()
    a=scaler.fit_transform(a);b=scaler.transform(b);model=module.build_model(name)
    t=perf_counter();model.fit(a,np.repeat(y[tr],19));seconds=perf_counter()-t
    prediction=predict(model,b);old=np.load(P/f'capacity_v4/inner_0_0_{name}.npz',allow_pickle=False)
    np.testing.assert_array_equal(prediction,old['prediction'])
    if name.startswith('XGB'):assert json.loads(model.get_booster().save_config())['learner']['generic_param']['device']=='cuda:0'
    joblib.dump(dict(scaler=scaler,model=model),dest)
    rows.append(dict(model=name,fit_seconds=seconds,model_bytes=dest.stat().st_size,train_trials=len(tr),evaluation_trials=len(te),
        original_prediction_exact=True,holdout_used=False,purpose='cost reconstruction, no model selection'))
    pd.DataFrame(rows).to_csv(R/'cost_refits.csv',index=False);print(json.dumps(rows[-1]),flush=True)
