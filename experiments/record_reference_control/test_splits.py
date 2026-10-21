"""Unequal record lengths must not break class balance or record separation."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from record_splits import schedules


def test_unequal_records_have_exact_coverage_and_no_group_leakage():
    groups = np.repeat(np.arange(64), 20 + np.arange(64) % 9)
    labels = groups % 8
    trial = 10000 + np.arange(len(groups))
    lookup = {int(t): i for i, t in enumerate(trial)}
    schedule = schedules(trial, labels, groups)
    assert schedule == schedules(trial, labels, groups)
    for protocol in ['trial_shared', 'record_disjoint']:
        evaluated = []
        for item in [s for s in schedule if s['protocol'] == protocol]:
            a = np.array([lookup[t] for t in item['train']])
            b = np.array([lookup[t] for t in item['evaluation']])
            assert not set(a) & set(b)
            assert set(a) | set(b) == set(range(len(trial)))
            train_records, eval_records = np.unique(groups[a]), np.unique(groups[b])
            if protocol == 'record_disjoint':
                assert not set(train_records) & set(eval_records)
                np.testing.assert_array_equal(np.bincount(train_records % 8), np.repeat(6, 8))
                np.testing.assert_array_equal(np.bincount(eval_records % 8), np.repeat(2, 8))
            else:
                assert len(set(train_records) & set(eval_records)) == 64
            evaluated.extend(item['evaluation'])
        assert sorted(evaluated) == sorted(trial)
