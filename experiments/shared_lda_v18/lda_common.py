"""Existing inputs with a fixed shared-covariance classifier."""
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from semg.paths import SOURCE_ROOT,RUN_ROOT
sys.path.append(str(SOURCE_ROOT/'experiments/shrink_qda_v17'))
from qda_data import inputs,baseline
from gaussian_boundary import to_eight
P=RUN_ROOT/'private/shared_lda_v18';R=RUN_ROOT/'results/shared_lda_v18'


def make_lda():
    return LinearDiscriminantAnalysis(solver='lsqr',shrinkage=None,covariance_estimator=LedoitWolf(store_precision=False,assume_centered=False))
