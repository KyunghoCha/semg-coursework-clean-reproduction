"""Existing clean-project inputs and frozen v25 component provenance."""
import json, os, sys
from semg.paths import RUN_ROOT, SOURCE_ROOT
sys.path.insert(0, str(SOURCE_ROOT/'experiments/mixture_weights_v25'))
from sources import inputs, source_paths, load, snapshot, check_hashes, digest
from weights import combine
sys.path.insert(0, str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import configure_audit, make_model, trial_cv, array_hash
from tuning_rules import CANDIDATES, SIZE_CAP, choose
EXPERIMENT = os.environ.get('SEMG_SVM_COMPARISON', 'svm_tuning_v29')
if EXPERIMENT == 'svm_regularization_v30':
    CANDIDATES = {'C1': dict(C=1., gamma='scale')}
elif EXPERIMENT == 'svm_metric_v31':
    CANDIDATES = {'whitened': dict(C=100., gamma='scale')}
    sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm_metric_v31'))
    from covariance_metric import ShrinkageWhitening
elif EXPERIMENT != 'svm_tuning_v29':
    raise ValueError('Unregistered SVM experiment')
P = RUN_ROOT/'private'/EXPERIMENT
R = RUN_ROOT/'results'/EXPERIMENT


def betas():
    return json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas']


def model_path(stage, fold, candidate):
    return source_paths(stage, fold)[2].with_suffix('.joblib') if candidate == 'incumbent' else P/f'{stage}_{candidate}_{fold}.joblib'
