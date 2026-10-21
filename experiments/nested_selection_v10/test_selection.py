from selection_rule import select

def test_selection_uses_inner_count_then_fixed_ties():
    rows=[dict(model='fine64',correct=100,features=905,rounds=400),dict(model='narrow64',correct=100,features=410,rounds=400),dict(model='direct8',correct=99,features=905,rounds=3200)]
    assert select(rows)=='narrow64'
    rows[2]['correct']=101
    assert select(rows)=='direct8'
