import pandas as pd
from semg.paths import RUN_ROOT
R=RUN_ROOT/'results/line_projection_v11';V=RUN_ROOT/'results/fine_spectrum_v8'
assert (R/'verification.json').exists()
a=pd.concat([pd.read_csv(V/'scores.csv'),pd.read_csv(R/'scores.csv')],ignore_index=True)
a.groupby('model').agg(accuracy=('accuracy','mean'),std=('accuracy',lambda x:x.std(ddof=0)),model_mb=('model_megabytes','mean'),fit_seconds=('fit_seconds','mean')).to_csv(R/'comparison_summary.csv')
pd.concat([pd.read_csv(V/'pooled.csv'),pd.read_csv(R/'pooled.csv')],ignore_index=True).to_csv(R/'comparison.csv',index=False)
print(a[['model','fold','accuracy','model_megabytes','fit_seconds']].to_string(index=False))
