"""CPU cost of fixed fold-0 AR controls and the selected outer model."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import time_features
from ar_features import ar_features
from run import features,P,R,source_base
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as rich_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve measurement')
    before=idle();rows=pd.read_csv(R/'scores.csv')
    selected=json.loads((R/'selection.json').read_text())['choices']
    cases=[(stage,name,0) for name in ['td_ar','td_ar_res'] for stage in ['trial_shared','record_disjoint']]
    for k,name in enumerate(selected):
        if name!='narrow64':cases.extend([('outer',name,k),('outer','narrow64',k)])
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());dev=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    assert len(ids['trial_ids'])==256 and set(ids['trial_ids'])<=dev
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    reports=[];raws={}
    for stage,name,fold in cases:
        stem=f'{stage}_{name}_{fold}';path=source_base(stage,fold).with_suffix('.joblib') if name=='narrow64' else P/f'{stem}.joblib';digest=hash_file(path);obj=joblib.load(path);model=obj['model'];scaler=obj['scaler']
        model.set_params(device='cpu',n_jobs=6)
        def feature(v):
            base=time_features(v) if name.startswith('td') else rich_feature(v)[:,COLS]
            if name=='narrow64':return base
            ar=ar_features(v)
            return np.c_[base,ar if name.endswith('_res') else ar[:,:30]]
        def infer(v):return model.get_booster().inplace_predict(scaler.transform(v).astype(np.float32))
        cache=features('narrow_ar')[:,:,:410] if name=='narrow64' else features(name);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);actual=feature(w)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(infer(actual).argmax(1),infer(expected).argmax(1))
        z=np.load(path.with_suffix('.npz'));ev=z['trial'][::19];q=infer(cache[ev].reshape(-1,cache.shape[-1]))
        np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5)
        q8=q.reshape(-1,8,8).sum(1) if q.shape[1]==64 else q
        np.testing.assert_array_equal(q8.argmax(1),z['prediction'])
        timing={};raws[stem]={}
        for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('scaler_model_probability_batch1',lambda i:infer(actual[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
            for i in range(20):fn(i)
            times=[]
            for i in range(200):
                start=perf_counter_ns();fn(i);times.append((perf_counter_ns()-start)/1e6)
            timing[label]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),warmup=20,repeats=200);raws[stem][label]=times
        if name=='narrow64':
            source=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/outer.csv')
            part=source[(source.model==name)&(source.fold==fold)]
            if part.empty:
                source=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');part=source[(source.model==name)&(source.fold==fold)&(source.stage==stage)]
            row=part.iloc[0];classes=int(row.learning_classes);trees=int(row.trees)
        else:
            row=rows[(rows.stage==stage)&(rows.model==name)&(rows.fold==fold)].iloc[0];classes=int(row.classes);trees=int(row.trees)
        assert hash_file(path)==digest
        reports.append(dict(case=stem,utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6,features=int(row.features),classes=classes,trees=trees,representative_fold=fold,training_trials=int(row.train_trials),model_bytes=path.stat().st_size,fit_seconds=float(row.fit_seconds),timing=timing,
                            model_sha256=digest,model_unchanged=True,restored_evaluation_windows=len(q),restored_predictions_equal=True,feature_cache_matches=True,signal_acquisition_excluded=True,model_loading_excluded=True))
        print(json.dumps(reports[-1]),flush=True)
    after=idle()
    (P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    (P/'cost_raw_timings.json').open('x').write(json.dumps(raws))
    (R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
