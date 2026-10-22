"""Same clean-project features, grouped splits, and fixed LDA/SVM probabilities."""
import hashlib,json,sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import inputs,baseline
P=RUN_ROOT/'private/regularized_boosting_v24'
R=RUN_ROOT/'results/regularized_boosting_v24'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.view(np.uint8));return h.hexdigest()


def paths(stage,fold):
    lda='shared_lda_v18' if stage=='inner' else 'lda_blend_v19'
    svm='svm410_v20' if stage=='inner' else 'svm_blend_v21'
    return [baseline(stage,fold),RUN_ROOT/f'private/{lda}/{stage}_lda_{fold}.npz',RUN_ROOT/f'private/{svm}/{stage}_svm_{fold}.npz']


def components(stage,fold,ids,y):
    zs=[np.load(path) for path in paths(stage,fold)]
    for z in zs:
        np.testing.assert_array_equal(z['trial'],np.repeat(ids,19))
        np.testing.assert_array_equal(z['truth'],np.repeat(y[ids],19))
        assert z['probability'].shape==(len(ids)*19,64)
    return [z['probability'] for z in zs]


def mix(xgb,lda,svm):
    parts=[a.reshape(-1,8,8).sum(1) for a in [xgb,lda,svm]]
    return .8*(.8*parts[0]+.2*parts[1])+.2*parts[2]


def snapshot(paths_):return {str(p.relative_to(RUN_ROOT)):digest(p) for p in paths_}


def check_hashes(hashes):
    for name,sha in hashes.items():assert digest(RUN_ROOT/name)==sha,name
