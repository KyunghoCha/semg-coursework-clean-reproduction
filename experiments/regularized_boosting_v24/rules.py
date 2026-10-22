"""Frozen candidates and an inner-only selection rule."""
SETTINGS = {
    'reg400': dict(n_estimators=400,max_depth=6,min_child_weight=5.,reg_lambda=5.),
    'shallow800': dict(n_estimators=800,max_depth=4,min_child_weight=1.,reg_lambda=1.),
}
NAMES = ('incumbent', *SETTINGS)


def select(rows):
    assert {r['candidate'] for r in rows} == set(NAMES)
    baseline = next(r for r in rows if r['candidate']=='incumbent')
    allowed = [r for r in rows if r['xgb_model_bytes'] <= 2*baseline['xgb_model_bytes']]
    return min(allowed, key=lambda r: (-r['correct'], r['candidate']!='incumbent', r['xgb_model_bytes'], NAMES.index(r['candidate'])))['candidate']
