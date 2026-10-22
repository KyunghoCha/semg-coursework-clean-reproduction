"""Fixed clean-project inputs and the already selected v19 reference."""
import json,sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.append(str(SOURCE_ROOT/'experiments/shared_lda_v18'))
from lda_common import inputs,baseline,to_eight
P=RUN_ROOT/'private/svm410_v20';R=RUN_ROOT/'results/svm410_v20'


def reference(stage,k):
    if stage=='outer':
        z=np.load(RUN_ROOT/f'private/lda_blend_v19/outer_selected_{k}.npz')
        return dict(truth=z['truth'],trial=z['trial'],probability=z['probability'],prediction=z['prediction'])
    z=np.load(baseline('inner',k));l=np.load(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{k}.npz');np.testing.assert_array_equal(z['trial'],l['trial']);np.testing.assert_array_equal(z['truth'],l['truth'])
    alpha=json.loads((RUN_ROOT/'results/lda_blend_v19/selection.json').read_text())['fold_alphas'][k];q=(1-alpha)*to_eight(z['probability'])+alpha*to_eight(l['probability'])
    return dict(truth=z['truth'],trial=z['trial'],probability=q,prediction=q.argmax(1))
