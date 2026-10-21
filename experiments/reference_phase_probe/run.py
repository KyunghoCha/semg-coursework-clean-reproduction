"""Early-reference spectral diagnostics with frozen OOF models; no training."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.data import CHANNELS,marker_indices,extract_segments,parse_name
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import cross_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features
P=RUN_ROOT/'private';R=RUN_ROOT/'results/reference_phase_probe'

def features(w):return np.c_[window_blocks(w)['rich'],cross_features(w),fine_features(w)]

def main(raw):
    if (R/'complete.json').exists():raise FileExistsError('Preserve completed probe')
    R.mkdir(parents=True,exist_ok=True)
    split=json.loads((P/'splits.json').read_text());dev=np.array(split['train']);held=set(split['test']);meta=pd.read_csv(P/'trials.csv');inventory=pd.read_csv(RUN_ROOT/'results/input_manifest.csv').set_index('file')
    mapping={int(t):j for j,t in enumerate(dev)};early=np.empty((len(dev),200,5));files=sorted(Path(raw).glob('*.csv'));assert len(files)==64
    for record,path in enumerate(files):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==inventory.loc[path.name,'sha256']
        frame=pd.read_csv(path);marks=marker_indices(frame);part=meta.iloc[record*30:(record+1)*30]
        np.testing.assert_array_equal(marks,part.mark);assert part.recording.eq(path.stem).all()
        assert part.label.eq(parse_name(path)[3]).all()
        signal=frame[CHANNELS].to_numpy(dtype=float);block=extract_segments(signal,marks,start=0,length=200)
        for j in range(30):
            trial=record*30+j
            if trial in mapping:early[mapping[trial]]=block[j]
    late=np.load(P/'cache/segments.npy',mmap_mode='r')[dev,-200:,:]
    stats=[];variance=[];group=meta.iloc[dev].recording.to_numpy()
    for phase,signal in [('early_0_0.2',early),('late_3.3_3.5',late)]:
        f,power=periodogram(signal,fs=1000,window='hann',detrend='constant',axis=1);power*=5
        rms=np.sqrt((signal**2).mean(axis=1));total=power.sum(axis=1)
        for c,name in enumerate(['ECR','ED','FDS','Biceps','Triceps']):
            for hz in [300,420]:
                a=power[:,f==hz,c][:,0];ratio=a/np.maximum(total[:,c],1e-12)
                stats.append(dict(phase=phase,channel=name,frequency_hz=hz,trials=len(dev),rms_median=float(np.median(rms[:,c])),power_median=float(np.median(a)),relative_power_median=float(np.median(ratio))))
                value=np.log10(np.maximum(a,1e-12));mu=value.mean();between=within=0.
                for record in np.unique(group):
                    v=value[group==record];between+=len(v)*(v.mean()-mu)**2;within+=((v-v.mean())**2).sum()
                totalvar=((value-mu)**2).sum();assert abs(between+within-totalvar)<1e-8
                variance.append(dict(phase=phase,channel=name,frequency_hz=hz,records=64,between_record_fraction=float(between/totalvar),within_record_fraction=float(within/totalvar)))
    x=features(early);xl=features(late);rows=[];covered=[]
    evidence=P/'reference_phase_probe';evidence.mkdir(parents=True,exist_ok=True)
    if (evidence/'inputs.npz').exists():raise FileExistsError('Preserve diagnostic evidence')
    np.savez(evidence/'inputs.npz',trial=dev,early=early,late=late,early_features=x,late_features=xl)
    schedule=json.loads((P/'fine_spectrum_v8/splits.json').read_text())
    for fold,s in enumerate(schedule):
        ids=np.array(s['evaluation']);assert not set(ids)&held;ix=np.array([mapping[int(i)] for i in ids]);covered.extend(ids)
        saved=joblib.load(P/f'fine_spectrum_v8/fine400_{fold}.joblib');model=saved['model'];model.set_params(device='cpu',n_jobs=1)
        truth=meta.iloc[ids].label.to_numpy();old=np.load(P/f'fine_spectrum_v8/fine400_{fold}.npz')['prediction'].reshape(-1,19)[:,-1]
        for phase,value in [('early_0_0.2',x[ix]),('late_3.3_3.5',xl[ix])]:
            prob=model.get_booster().inplace_predict(saved['scaler'].transform(value).astype(np.float32));pred=prob.reshape(-1,8,8).sum(axis=1).argmax(axis=1)
            if phase=='late_3.3_3.5':np.testing.assert_array_equal(pred,old)
            np.savez(evidence/f'{phase}_{fold}.npz',trial=ids,truth=truth,prediction=pred,probability=prob)
            rows.append(dict(phase=phase,fold=fold,trials=len(ids),correct=int((pred==truth).sum()),**manual_metrics(truth,pred)))
    assert sorted(covered)==sorted(dev)
    pd.DataFrame(stats).to_csv(R/'power_summary.csv',index=False);pd.DataFrame(variance).to_csv(R/'variance_summary.csv',index=False);pd.DataFrame(rows).to_csv(R/'scores.csv',index=False)
    summary=pd.DataFrame(rows).groupby('phase').agg(trials=('trials','sum'),correct=('correct','sum'),accuracy=('accuracy','mean'),std=('accuracy',lambda x:x.std(ddof=0)));summary.to_csv(R/'summary.csv')
    (R/'complete.json').write_text(json.dumps(dict(raw_hashes_rechecked=64,development_trials=1536,early_signal_ms=200,late_signal_ms=200,frozen_oof_models=3,late_predictions_match_saved=True,new_fits=0,historical_test_predictions=0,physiological_rest_verified=False,origin_of_frequency_components_confirmed=False,threads=1),indent=2)+'\n')
    print(summary.to_string());print(pd.DataFrame(variance).groupby(['phase','frequency_hz']).between_record_fraction.mean().to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main(sys.argv[1])
