"""Fixed forest settings and deterministic inner selection."""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

NAMES = ['fine64', 'narrow64', 'RF100_fine', 'ET400_fine']
NEW = ['RF100_fine', 'ET400_fine']


def make_forest(name):
    common = dict(criterion='gini', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, class_weight=None, ccp_alpha=0.,
                  random_state=42, n_jobs=6, verbose=1)
    if name == 'RF100_fine':
        return RandomForestClassifier(n_estimators=100, max_features='sqrt', bootstrap=True, **common)
    if name == 'ET400_fine':
        return ExtraTreesClassifier(n_estimators=400, max_features=.5, bootstrap=False, **common)
    raise ValueError(name)


def select(rows):
    if len(rows) != 4 or {r['model'] for r in rows} != set(NAMES):
        raise ValueError('Expected all four registered candidates')
    return min(rows, key=lambda r: (-r['correct'], r['model_bytes'], NAMES.index(r['model'])))['model']
