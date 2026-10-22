import json
import numpy as np
from sklearn.base import clone
from audited_svm import trial_cv,make_model,configure_audit


def test_trial_grouping_and_all_classes():
    y=np.tile(np.arange(8),6);cv=trial_cv(y,91)
    for a,b in cv:
        assert not set(a//19)&set(b//19)
        for index in [a,b]:
            _,counts=np.unique(index//19,return_counts=True);np.testing.assert_array_equal(counts,np.full(len(counts),19))
        np.testing.assert_array_equal(np.unique(np.repeat(y,19)[a]),np.arange(8))


def test_calibration_retains_three_oof_fits_and_one_full_fit(tmp_path):
    rng=np.random.default_rng(5);target=np.tile(np.arange(3),6);x=rng.normal(size=(18*19,4))+np.repeat(target,19)[:,None]
    configure_audit(tmp_path);m=clone(make_model(trial_cv(target,19)));m.fit(x,np.repeat(target,19));assert len(m.calibrated_classifiers_)==1
    logs=[json.loads(p.read_text()) for p in sorted(tmp_path.glob('*.json'))];assert len(logs)==4 and logs[-1]['windows']==342
    assert [r['windows'] for r in logs[:3]]==[228]*3 and all(r['fit_status']==0 for r in logs)
    q=m.predict_proba(x[:5]);assert q.shape==(5,3);np.testing.assert_allclose(q.sum(1),1)
