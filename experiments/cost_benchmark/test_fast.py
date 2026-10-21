import sys
import numpy as np
from semg.paths import SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import cross_features
from features_fast import fast_cross_features


def test_fast_cross_equivalence_and_zero():
    rng=np.random.default_rng(17);w=rng.normal(size=(7,200,5))
    w[1,:,2]=0;w[2,:,3]=w[2,:,1];w[3]=0
    np.testing.assert_allclose(fast_cross_features(w),cross_features(w),atol=1e-14,rtol=1e-13)
    np.testing.assert_array_equal(fast_cross_features(w)[3],np.zeros(160))
