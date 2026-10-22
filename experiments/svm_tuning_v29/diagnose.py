"""Inspect only existing inner predictions and model parameters before new fits."""
import json
import joblib
import numpy as np
from common29 import P, R, inputs, source_paths, load, snapshot, betas, combine


def main():
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    x, y, target, schedule = inputs(); rows = []; hashes = {}
    for k, s in enumerate(schedule):
        paths = source_paths('inner', k); hashes.update(snapshot(paths + [paths[2].with_suffix('.joblib')]))
        m = joblib.load(paths[2].with_suffix('.joblib')).calibrated_classifiers_[0].estimator
        svc, sc = m.named_steps['svc'], m.named_steps['scale']
        ids = np.array(s['inner_evaluation']); qs = load('inner', k, ids, y)
        q = combine(*qs, betas()[k]); t = np.repeat(y[ids], 19); pred = q.argmax(1)
        sv = qs[2].reshape(-1, 8, 8).sum(1); sp = sv.argmax(1); correct = pred == t; scorrect = sp == t
        rows.append(dict(fold=k, C=svc.C, gamma=svc.gamma, effective_gamma=svc._gamma,
                         constant_columns=int((sc.var_ == 0).sum()), support_vectors=len(svc.support_),
                         train_windows=int(sc.n_samples_seen_), support_fraction=len(svc.support_)/int(sc.n_samples_seen_),
                         mixture_correct=int(correct.sum()), mixture_wrong=int((~correct).sum()),
                         svm_correct_when_mixture_wrong=int((scorrect & ~correct).sum()),
                         mixture_wrong_with_truth_in_top2=int(((np.argsort(q, axis=1)[:, -2:] == t[:, None]).any(1) & ~correct).sum()),
                         wrong_by_class=np.bincount(t[~correct], minlength=8).tolist()))
    (R/'prior_inner_diagnostic.json').open('x').write(json.dumps(dict(rows=rows, source_hashes=hashes, outer_predictions_read=False), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
