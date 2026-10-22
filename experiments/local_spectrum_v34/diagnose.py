"""Development-only representation checks, without any model training."""
import json, sys
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import pandas as pd
from semg.paths import SOURCE_ROOT, RUN_ROOT
from semg.features import windows
from features_tf34 import local_band_features
sys.path.insert(0, str(SOURCE_ROOT / 'experiments/shrink_qda_v17'))
from qda_data import inputs
sys.path.insert(0, str(SOURCE_ROOT / 'experiments/regularized_boosting_v24'))
from data import digest

P = RUN_ROOT / 'private/local_spectrum_v34'
R = RUN_ROOT / 'results/local_spectrum_v34'


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n')


def main():
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    if (R / 'diagnostic.json').exists() or (P / 'features.npz').exists():
        raise FileExistsError('Inspect existing diagnostic first')
    split = json.loads((RUN_ROOT / 'private/splits.json').read_text())
    dev = np.array(split['train']); held = set(split['test'])
    segments_path = RUN_ROOT / 'private/cache/segments.npy'
    source_hash = digest(segments_path)
    segments = np.load(segments_path, mmap_mode='r')
    start = perf_counter()
    added = np.stack([local_band_features(windows(segments[i])) for i in dev])
    elapsed = perf_counter() - start
    assert added.shape == (1536, 19, 80) and not set(dev) & held
    np.savez(P / 'features.npz', trial=dev, features=added)
    lookup = {int(v): i for i, v in enumerate(dev)}
    _, y, target, schedule = inputs()
    old = np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                          np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                          np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy'),
                          np.load(RUN_ROOT/'private/ar_residual_v15/features.npy')], axis=2)
    assert old.shape == (1920, 19, 940)
    rows = []
    for k, s in enumerate(schedule):
        tr, te = np.array(s['inner_train']), np.array(s['inner_evaluation'])
        assert not set(tr) & set(te) and not (set(tr) | set(te)) & held
        a = old[tr].reshape(-1, 940); b = added[[lookup[int(v)] for v in tr]].reshape(-1, 80)
        a = a - a.mean(0); b = b - b.mean(0)
        a /= np.maximum(np.linalg.norm(a, axis=0), 1e-15)
        b /= np.maximum(np.linalg.norm(b, axis=0), 1e-15)
        nearest = np.max(np.abs(a.T @ b), axis=0)
        new = added[[lookup[int(v)] for v in te]].reshape(-1, 4, 4, 5)
        variation = np.abs(np.diff(new, axis=1)).mean(axis=(1, 2, 3))
        w = np.concatenate([windows(segments[i]) for i in te])
        rms = np.sqrt(np.mean(w*w, axis=(1, 2)))
        sub = np.sqrt(np.mean(w.reshape(-1, 5, 40, 5)**2, axis=(2, 3)))
        broad = sub.std(axis=1) / np.maximum(sub.mean(axis=1), 1e-15)
        q = np.load(RUN_ROOT/f'private/regularized_boosting_v24/inner_shallow800_{k}.npz')
        np.testing.assert_array_equal(q['trial'], np.repeat(te, 19))
        truth = np.repeat(y[te], 19)
        np.testing.assert_array_equal(q['truth'], truth)
        prob = q['probability'].reshape(-1, 8, 8).sum(1)
        error = (prob.argmax(1) != truth).astype(float)
        # 64 recording/target indicators already control the eight classes.
        # Nuisance regression is a descriptive diagnostic, never an inference input.
        design = np.c_[np.eye(64)[np.repeat(target[te], 19)],
                       np.log(np.maximum(rms, 1e-15)), broad]
        residual = np.c_[variation, error] - design @ np.linalg.lstsq(design, np.c_[variation, error], rcond=None)[0]
        corr = float(np.corrcoef(residual.T)[0, 1])
        rows.append(dict(fold=k, training_windows=len(a), evaluation_windows=len(w),
                         nearest_existing_abs_correlation_min=float(nearest.min()),
                         nearest_existing_abs_correlation_median=float(np.median(nearest)),
                         nearest_existing_abs_correlation_max=float(nearest.max()),
                         near_duplicate_columns_0999=int((nearest >= .999).sum()),
                         conditional_variation_error_correlation=corr,
                         raw_variation_error_correlation=float(np.corrcoef(variation,error)[0,1]),
                         mean_variation_correct=float(variation[error == 0].mean()),
                         mean_variation_error=float(variation[error == 1].mean())))
    assert source_hash == digest(segments_path)
    report = dict(utc=datetime.now(timezone.utc).isoformat(), development_trials=1536,
                  historical_test_trials_processed=0, windows_per_trial=19, features=80,
                  combined_features=490, existing_features_checked=940,
                  generation_seconds=elapsed, rows=rows,
                  learned_transforms=0, no_independent_window_significance_test=True,
                  conditional_controls=['recording (also class)', 'log broadband RMS', '40ms broadband RMS variability'],
                  source_sha256=source_hash, features_sha256=digest(P/'features.npz'),
                  feature_code_sha256=digest(SOURCE_ROOT/'experiments/local_spectrum_v34/features_tf34.py'))
    save(R/'diagnostic.json', report)
    pd.DataFrame(rows).to_csv(R/'diagnostic.csv', index=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
