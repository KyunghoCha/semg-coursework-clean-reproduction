"""Fixed CPU comparison of the one QDA candidate and matching XGB; no fitting."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from gaussian_boundary import to_eight
from qda_data import P,R,inputs,baseline
sys.path.append(str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve measurement')
    before=idle();cache,_,_,_=inputs();selection=json.loads((R/'selection.json').read_text())['choices'];pairs=[('inner',0)]
    if 'qda' in selection:pairs.append(('outer',selection.index('qda')))
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());assert len(ids['trial_ids'])==256
    assert set(ids['trial_ids'])<=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    def feature(v):return full_feature(v)[:,COLS]
    actual=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    reports=[];raws={}
    for stage,k in pairs:
        for name in ['narrow64','qda']:
            stem=f'{stage}_{name}_{k}';path=baseline(stage,k).with_suffix('.joblib') if name=='narrow64' else P/f'{stage}_qda_{k}.joblib';digest=hash_file(path);obj=joblib.load(path);m=obj['model'];sc=obj['scaler']
            if name=='narrow64':m.set_params(device='cpu',n_jobs=6)
            def infer(v):
                a=sc.transform(v);q=m.get_booster().inplace_predict(a.astype(np.float32)) if name=='narrow64' else m.predict_proba(a)
                return to_eight(q)
            z=np.load(path.with_suffix('.npz'));te=z['trial'][::19];np.testing.assert_allclose(infer(cache[te].reshape(-1,410)),to_eight(z['probability']),atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(infer(actual).argmax(1),infer(expected).argmax(1))
            timing={};raws[stem]={}
            for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('scaler_model_probability_batch1',lambda i:infer(actual[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
                for i in range(20):fn(i)
                values=[]
                for i in range(200):
                    start=perf_counter_ns();fn(i);values.append((perf_counter_ns()-start)/1e6)
                timing[label]=dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),warmup=20,repeats=200);raws[stem][label]=values
            if name=='qda':table=pd.read_csv(R/'scores.csv');row=table[(table.stage==stage)&(table.fold==k)].iloc[0]
            else:
                table=pd.read_csv(RUN_ROOT/f'results/nested_selection_v10/{stage}.csv');part=table[(table.model==name)&(table.fold==k)]
                if part.empty:
                    table=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');part=table[(table.model==name)&(table.fold==k)&(table.stage==stage)]
                row=part.iloc[0]
            assert hash_file(path)==digest
            report=dict(case=stem,utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6,features=410,learning_targets=64,training_trials=int(row.train_trials),model_bytes=path.stat().st_size,fit_seconds=float(row.fit_seconds),fit_device='cpu' if name=='qda' else 'cuda',timing=timing,model_sha256=digest,model_preserved=True,feature_cache_equal=True,restored_probability_equal=True,signal_acquisition_excluded=True,model_loading_excluded=True)
            reports.append(report);print(json.dumps(report),flush=True)
    after=idle();(P/'cost_raw_timings.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    (R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
