"""A fixed SVC pipeline with retained base fits for calibration auditing."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_AUDIT_DIRECTORY=None


def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()


def configure_audit(path):
    global _AUDIT_DIRECTORY
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if list(path.iterdir()):raise FileExistsError('Preserve base-fit audit artifacts')
    _AUDIT_DIRECTORY=path


class AuditedPipeline(Pipeline):
    def fit(self,x,y=None,**params):
        if _AUDIT_DIRECTORY is None:raise RuntimeError('Configure private audit destination first')
        index=len(list(_AUDIT_DIRECTORY.glob('base_*.json')));stem=f'base_{index:02d}'
        print('SVC_START',_AUDIT_DIRECTORY.name,stem,len(x),datetime.now(timezone.utc).isoformat(),flush=True)
        start=perf_counter();super().fit(x,y,**params);seconds=perf_counter()-start;svc=self.named_steps['svc']
        assert svc.fit_status_==0
        path=_AUDIT_DIRECTORY/f'{stem}.joblib';assert not path.exists();joblib.dump(self,path)
        info=dict(index=index,utc=datetime.now(timezone.utc).isoformat(),windows=len(x),features=x.shape[1],x_sha256=array_hash(x),y_sha256=array_hash(y),fit_seconds=seconds,classes=svc.classes_.tolist(),fit_status=int(svc.fit_status_),support_vectors=int(len(svc.support_vectors_)),maximum_pair_iterations=int(svc.n_iter_.max()),model_bytes=path.stat().st_size,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (_AUDIT_DIRECTORY/f'{stem}.json').open('x').write(json.dumps(info,indent=2)+'\n');print('SVC_DONE',_AUDIT_DIRECTORY.name,stem,seconds,flush=True)
        return self


def trial_cv(target,seed):
    target=np.asarray(target);splits=[]
    for a,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(target)),target):
        assert not set(a)&set(b) and np.array_equal(np.unique(target[a]),np.unique(target))
        splits.append(((a[:,None]*19+np.arange(19)).ravel(),(b[:,None]*19+np.arange(19)).ravel()))
    assert sorted(np.concatenate([b for a,b in splits]))==list(range(len(target)*19))
    return splits


def make_model(cv):
    pipeline=AuditedPipeline([('scale',StandardScaler()),('svc',SVC(C=100.,kernel='rbf',gamma='scale',probability=False,cache_size=1024.,decision_function_shape='ovr',random_state=42,max_iter=-1,tol=.001))])
    return CalibratedClassifierCV(pipeline,method='sigmoid',cv=cv,n_jobs=1,ensemble=False)
