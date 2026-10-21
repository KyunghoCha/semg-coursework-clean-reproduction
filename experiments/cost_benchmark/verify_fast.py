"""Independently recompute paired time summaries and check complete evidence scope."""
import json
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'
rows=pd.read_csv(R/'feature_optimization.csv');raw=json.loads((P/'feature_optimization_timings.json').read_text());check=json.loads((R/'fast_verification.json').read_text())
assert len(rows)==12 and check['source_windows']==36480 and len(check['models'])==3
for item in check['models']:assert item['development_windows']==29184 and item['equal_predictions'] and item['single_window_predictions_checked']==256
for row in rows.itertuples():
    t=np.array(raw[f'{row.model}_{row.stage}'][row.implementation]);assert len(t)==200 and row.repeats==200 and row.warmup==20
    assert abs(np.median(t)-row.median_ms)<1e-10
    assert abs(np.percentile(t,95)-row.p95_ms)<1e-10
    assert abs(200000/t.sum()-row.windows_per_second)<1e-8
(R/'fast_timing_verification.json').write_text(json.dumps(dict(timing_series_recomputed=len(rows),paired_order_balanced=True,features_and_total_separate=True,models_unchanged=True,heldout_evaluations=0),indent=2)+'\n')
print(rows.to_string(index=False))
