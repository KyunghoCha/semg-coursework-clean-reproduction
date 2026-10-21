"""Recompute original and in-place API latency summaries without retraining."""
import json
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'
assert (R/'inplace_complete.json').exists()
rows=[];series=0
names=['lecture_XGB400','cross64_400_v7','cross64_800_v7']
if (RUN_ROOT/'results/fine_spectrum_v8/verification.json').exists():names.append('fine64_400_v8')
for model in names:
    for device in ['cpu','cuda']:
        for api in ['dmatrix','inplace']:
            stem=f'{model}_{device}'+('_inplace' if api=='inplace' else '')
            z=json.loads((R/f'{stem}.json').read_text());raw=json.loads((P/f'{stem}_timings.json').read_text())
            for stage,v in raw.items():
                a=np.array(v);m=z['timing'][stage]
                assert len(a)==((10 if stage.endswith('batch256') else 100) if api=='inplace' else (30 if stage.endswith('batch256') else 200))
                series+=1
                assert abs(np.median(a)-m['median_ms'])<1e-10 and abs(np.percentile(a,95)-m['p95_ms'])<1e-10
                assert abs(sum(v)-m['measured_total_ms'])<1e-8
            if api=='inplace':
                probabilities=np.load(P/f'{stem}_probabilities.npz',allow_pickle=False)
                np.testing.assert_allclose(probabilities['direct'],probabilities['ordinary'],atol=1e-6,rtol=1e-5)
                assert float(np.max(abs(probabilities['direct']-probabilities['ordinary'])))==z['maximum_probability_difference']
                np.testing.assert_array_equal(np.load(P/f'{stem}_prediction.npy'),np.load(P/f'{model}_{device}_prediction.npy'))
            rows.append(dict(model=model,device=device,api=api,model_mb=z['model_bytes_with_scaler']/1e6,
                model_median_ms=z['timing']['model_batch1']['median_ms'],model_p95_ms=z['timing']['model_batch1']['p95_ms'],
                total_median_ms=z['timing']['total_batch1']['median_ms'],total_p95_ms=z['timing']['total_batch1']['p95_ms'],
                total_batch256_windows_per_second=z['timing']['total_batch256']['windows_per_second'],**z['memory']))
frame=pd.DataFrame(rows);frame.to_csv(R/'api_comparison.csv',index=False)
(R/'inplace_verification.json').write_text(json.dumps(dict(comparisons=len(rows),timing_series_recomputed=series,unchanged_256_predictions=True,
    probability_tolerance=dict(atol=1e-6,rtol=1e-5),training_or_test_evaluations=0),indent=2)+'\n')
print(frame.to_string(index=False))
