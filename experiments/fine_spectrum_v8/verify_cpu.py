"""Verify CPU deployment against all saved GPU OOF predictions, without new fits."""
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private';R=RUN_ROOT/'results/fine_spectrum_v8'
assert (R/'verification.json').exists()
x=np.concatenate([np.load(P/'diagnostic_v2/features.npz')['rich'],np.load(P/'cross_spectrum_v6/features.npy'),np.load(P/'fine_spectrum_v8/features.npy')],axis=2)
splits=json.loads((P/'fine_spectrum_v8/splits.json').read_text());rows=[]
with threadpool_limits(limits=6):
    for fold,s in enumerate(splits):
        saved=joblib.load(P/f'fine_spectrum_v8/fine400_{fold}.joblib');model=saved['model'];model.set_params(device='cpu',n_jobs=6)
        z=np.load(P/f'fine_spectrum_v8/fine400_{fold}.npz')
        inputs=saved['scaler'].transform(x[s['evaluation']].reshape(-1,905)).astype(np.float32)
        probability=model.get_booster().inplace_predict(inputs)
        np.testing.assert_array_equal(probability.reshape(-1,8,8).sum(axis=1).argmax(axis=1),z['prediction'])
        rows.append(dict(fold=fold,windows=len(inputs),same_predictions=True,maximum_probability_difference=float(np.max(abs(probability-z['probability'])))))
(R/'cpu_prediction_verification.json').write_text(json.dumps(dict(folds=rows,windows=sum(r['windows'] for r in rows),cpu_thread_cap=6,training_and_test_evaluations=0,scope='all saved development out-of-fold window predictions'),indent=2)+'\n')
print(rows)
