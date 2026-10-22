"""Frozen v21 probabilities and chronology; no raw signal processing or new fit."""
import hashlib,json,sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import reference,to_eight,baseline
P=RUN_ROOT/'private/causal_probability_v23';R=RUN_ROOT/'results/causal_probability_v23'


def schedule():return json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text())


def load_source(stage,fold):
    s=schedule()[fold];ids=np.array(s[stage+'_evaluation']);split=json.loads((RUN_ROOT/'private/splits.json').read_text());assert set(ids)<=set(split['train']) and not set(ids)&set(split['test']);y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];beta=json.loads((RUN_ROOT/'results/svm_blend_v21/selection.json').read_text())['fold_betas'][fold];assert beta==.2
    if stage=='inner':
        ref=reference(stage,fold);sv_path=RUN_ROOT/f'private/svm410_v20/inner_svm_{fold}.npz';sv=np.load(sv_path);np.testing.assert_array_equal(ref['trial'],sv['trial']);np.testing.assert_array_equal(ref['truth'],sv['truth']);q=(1-beta)*ref['probability']+beta*to_eight(sv['probability']);trial=sv['trial'];truth=sv['truth'];paths=[baseline(stage,fold),RUN_ROOT/f'private/shared_lda_v18/inner_lda_{fold}.npz',sv_path]
    else:
        path=RUN_ROOT/f'private/svm_blend_v21/outer_selected_{fold}.npz';z=np.load(path);q=z['probability'];trial=z['trial'];truth=z['truth'];np.testing.assert_array_equal(q.argmax(1),z['prediction']);paths=[path]
    np.testing.assert_array_equal(trial,np.repeat(ids,19));np.testing.assert_array_equal(truth,np.repeat(y[ids],19));assert q.shape==(len(ids)*19,8) and np.isfinite(q).all() and (q>=0).all();np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
    return q.astype(np.float64).reshape(-1,19,8),truth.reshape(-1,19),ids,paths


def snapshot(paths):return {str(p.relative_to(RUN_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def assert_preserved(hashes):
    for name,digest in hashes.items():assert hashlib.sha256((RUN_ROOT/name).read_bytes()).hexdigest()==digest
