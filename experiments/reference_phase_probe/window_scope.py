"""Verify dependence on supplied CSV samples within the current window only."""
import json
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import windows
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import cross_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features

def feature(w):return np.c_[window_blocks(w)['rich'],cross_features(w),fine_features(w)]

rng=np.random.default_rng(20261117);segment=rng.normal(size=(2000,5));base=feature(windows(segment));scaler=StandardScaler().fit(rng.normal(size=(100,905)))
for index in [0,8,18]:
    start=index*100;stop=start+200;changed=segment.copy()
    changed[:start]=rng.normal(100,20,size=changed[:start].shape);changed[stop:]=rng.normal(-200,30,size=changed[stop:].shape)
    altered=feature(windows(changed));np.testing.assert_array_equal(base[index],altered[index]);np.testing.assert_array_equal(scaler.transform(base[index:index+1]),scaler.transform(altered[index:index+1]))
    np.testing.assert_allclose(base[index:index+1],feature(segment[start:stop][None]),atol=1e-13,rtol=1e-12)
R=RUN_ROOT/'results/reference_phase_probe'
(R/'window_scope.json').write_text(json.dumps(dict(feature_dimensions=905,outside_window_perturbation_positions=[0,8,18],features_unchanged=True,fixed_scaler_outputs_unchanged=True,trial_statistics_used=False,current_feature_input='200 supplied CSV samples and 5 channels',supplied_csv_filter_causality_verified=False,raw_unfiltered_signal_available=False),indent=2)+'\n')
print('Current-window features unchanged after altering all outside-window samples')
