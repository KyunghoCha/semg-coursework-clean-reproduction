"""Resume only complete, independently checked rows; never silently replace artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from semg.verification import manual_metrics


def checked_rows(csv, private, keys, stem, suffixes):
    if not csv.exists():
        return {}
    rows = pd.read_csv(csv)
    assert not rows.duplicated(keys).any(), 'Duplicate checkpoint keys'
    result = {}
    for row in rows.to_dict('records'):
        name = stem(row)
        for suffix in suffixes(row):
            assert (private / (name + suffix)).is_file(), name + suffix
        z = np.load(private / (name + '.npz'), allow_pickle=False)
        for key, value in manual_metrics(z['truth'], z['prediction']).items():
            assert abs(value - row[key]) < 1e-12, (name, key)
        result[tuple(row[k] for k in keys)] = row
    return result


def write_identical_or_new(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Existing registration/split differs'
    else:
        path.write_text(json.dumps(value, indent=2))


def snapshot(private):
    path = private / 'recovery_existing_sha256.json'
    if not path.exists():
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in private.iterdir() if p.suffix in {'.npz', '.joblib', '.json'} and p != path}
        path.write_text(json.dumps(files, indent=2))


def verify_preserved(private):
    for name, expected in json.loads((private / 'recovery_existing_sha256.json').read_text()).items():
        assert hashlib.sha256((private / name).read_bytes()).hexdigest() == expected, name
