"""Read original CSVs and create new private arrays; never consumes older features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from semg.data import load_recordings, CLASSES
from semg.features import feature_dataset
from semg.evaluation import holdout, folds, check_split
from semg.paths import RUN_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    root = RUN_ROOT
    private = root/"private"
    if (root/'results/baseline.csv').exists():
        raise FileExistsError('Preserve completed results; prepare inputs in a fresh run directory')
    for directory in [private/'cache',private/'logs',root/'results/figures']:
        directory.mkdir(parents=True,exist_ok=True)
    segments, meta, inventory = load_recordings(args.raw)
    assert segments.shape == (1920, 2000, 5)
    meta.to_csv(private/"trials.csv", index=False)
    pd.DataFrame(inventory).to_csv(private/"input_manifest.csv", index=False)
    y = meta.label.to_numpy()
    tr, te = holdout(y)
    check_split(tr, te, len(y))
    split = dict(train=tr.tolist(), test=te.tolist(),
                 folds=[dict(train=a.tolist(), test=b.tolist()) for a,b in folds(y)])
    (private/"splits.json").write_text(json.dumps(split, indent=2))
    features = feature_dataset(segments)
    assert features.shape == (1920, 19, 25) and np.isfinite(features).all()
    np.save(private/"cache/segments.npy", segments)
    np.savez(private/"cache/baseline.npz", X=features, y=y)
    summary = dict(recordings=len(inventory), trials=len(y), windows=1920*19,
                   features=25, classes=CLASSES, trials_per_class=np.bincount(y).tolist(),
                   holdout_train_trials=len(tr), holdout_test_trials=len(te),
                   shared_holdout_trials=0, split_sha256=hashlib.sha256(
                       (private/"splits.json").read_bytes()).hexdigest(),
                   features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                   preprocessing="provided_bandpass_no_additional_filter")
    (root/"results/data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
