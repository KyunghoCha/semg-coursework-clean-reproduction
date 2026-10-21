"""CPU cost of one frozen gain control; no fit or model selection."""
import argparse,json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import time_features
from features import normalized_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['raw25','normalized25'],required=True);p.add_argument('--protocol',choices=['trial_shared','record_disjoint'],required=True);args=p.parse_args()
    stage='active_record_baseline' if args.kind=='raw25' else 'gain_invariance_v14'
    r=RUN_ROOT/'results/gain_invariance_v14';private=RUN_ROOT/'private/gain_invariance_v14';case=args.kind+'_'+args.protocol
    dest=r/f'cost_{case}.json'
    if dest.exists():raise FileExistsError('Preserve completed timing')
    assert (RUN_ROOT/f'results/{stage}/verification.json').exists()
    before=idle();path=RUN_ROOT/f'private/{stage}/{args.protocol}_0.joblib';original_hash=hash_file(path)
    obj=joblib.load(path);m=obj['model'];s=obj['scaler'];m.set_params(device='cpu',n_jobs=6)
    np.testing.assert_array_equal(m.classes_,np.arange(8))
    feature=time_features if args.kind=='raw25' else normalized_features
    cache=np.load(RUN_ROOT/'private/cache/baseline.npz')['X'] if args.kind=='raw25' else np.load(private/'features.npy',mmap_mode='r')
    z=np.load(path.with_suffix('.npz'));ev=z['trial'][::19];values=cache[ev].reshape(-1,25)
    def infer(x):return m.get_booster().inplace_predict(s.transform(x).astype(np.float32))
    q=infer(values);np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(q.argmax(axis=1),z['prediction'])
    assert json.loads(m.get_booster().save_config())['learner']['generic_param']['device']=='cpu'
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());dev=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);assert set(ids['trial_ids'])<=dev
    segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    x=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(x,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(infer(x).argmax(axis=1),infer(expected).argmax(axis=1))
    raw={};timing={}
    for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('scaler_model_probability_batch1',lambda i:infer(x[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
        for i in range(20):fn(i)
        times=[]
        for i in range(200):
            start=perf_counter_ns();fn(i);times.append((perf_counter_ns()-start)/1e6)
        raw[label]=times;timing[label]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),repeats=200,warmup=20)
    after=idle();assert hash_file(path)==original_hash
    row=pd.read_csv(RUN_ROOT/f'results/{stage}/scores.csv');row=row[(row.protocol==args.protocol)&(row.fold==0)].iloc[0]
    report=dict(case=case,utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6,features=25,classes=8,trees=3200,representative_fold=0,training_trials=int(row.train_trials),signal_ms=200,model_bytes=path.stat().st_size,fit_seconds=float(row.fit_seconds),timing=timing,model_sha256=original_hash,model_unchanged=True,restored_evaluation_windows=len(q),restored_predictions_equal=True,feature_cache_matches=True,signal_acquisition_excluded=True,model_loading_excluded=True)
    (private/f'cost_{case}_timings.json').open('x').write(json.dumps(raw))
    (private/f'cost_{case}_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    dest.open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
