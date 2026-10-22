from rules import select


def rows(scores=(10,11,12),sizes=(100,120,150)):
    return [dict(candidate=n,correct=c,xgb_model_bytes=s) for n,c,s in zip(['incumbent','reg400','shallow800'],scores,sizes)]


def test_selection_cost_limit_and_ties():
    assert select(rows())=='shallow800'
    assert select(rows(sizes=(100,120,201)))=='reg400'
    assert select(rows(scores=(12,12,12)))=='incumbent'
    assert select(rows(scores=(10,12,12),sizes=(100,150,120)))=='shallow800'
    assert select(rows(scores=(10,12,12),sizes=(100,120,120)))=='reg400'
