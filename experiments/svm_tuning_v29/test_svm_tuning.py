from tuning_rules import choose


def row(name, correct, size=100, sv=50):
    return dict(candidate=name, correct=correct, model_bytes=size, support_vectors=sv)


def test_incumbent_on_tie_and_size_cap():
    assert choose([row('incumbent', 90), row('broad', 90, 50), row('narrow', 95, 126)]) == 'incumbent'
    assert choose([row('incumbent', 90), row('broad', 91, 125)]) == 'broad'


def test_new_tie_is_deterministic():
    rows = [row('incumbent', 90), row('broad', 91, 120, 40), row('regularized', 91, 120, 30)]
    assert choose(rows) == choose(rows[::-1]) == 'regularized'


def test_experiment_output_isolation():
    """A follow-up must not write into the completed comparison's directories."""
    import json, os, subprocess, sys
    from pathlib import Path
    folder=str(Path(__file__).resolve().parent)
    script=("import sys,json;sys.path.insert(0,"+repr(folder)+");import common29 as c;"
            "print(json.dumps([c.P.name,c.R.name,list(c.CANDIDATES),c.model_path('inner',0,next(iter(c.CANDIDATES))).parent.name]))")
    env=dict(os.environ);env.pop('SEMG_SVM_COMPARISON',None)
    old=json.loads(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
    env['SEMG_SVM_COMPARISON']='svm_regularization_v30'
    new=json.loads(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
    assert old==['svm_tuning_v29','svm_tuning_v29',['regularized','broad','narrow'],'svm_tuning_v29']
    assert new==['svm_regularization_v30','svm_regularization_v30',['C1'],'svm_regularization_v30']
    env['SEMG_SVM_COMPARISON']='svm_metric_v31'
    metric=json.loads(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
    assert metric==['svm_metric_v31','svm_metric_v31',['whitened'],'svm_metric_v31']
    env['SEMG_SVM_COMPARISON']='../../unregistered'
    blocked=subprocess.run([sys.executable,'-c',script],env=env,text=True,capture_output=True)
    assert blocked.returncode!=0 and 'Unregistered SVM experiment' in blocked.stderr
