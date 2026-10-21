"""Independent count-based metric check, separate from the sklearn execution path."""
import numpy as np


def manual_metrics(truth,prediction):
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    if truth.shape!=prediction.shape or truth.ndim!=1 or not len(truth):raise ValueError('Expected equal nonempty vectors')
    if not np.isin(truth,np.arange(8)).all() or not np.isin(prediction,np.arange(8)).all():raise ValueError('Invalid class ID')
    matrix=np.zeros((8,8),dtype=np.int64)
    for a,b in zip(truth,prediction,strict=True):matrix[int(a),int(b)]+=1
    tp=np.diag(matrix);actual=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros(8),where=predicted!=0)
    recall=np.divide(tp,actual,out=np.zeros(8),where=actual!=0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(8),where=actual+predicted!=0)
    a,b=np.indices((8,8));n=int(matrix.sum())
    return dict(accuracy=float(tp.sum()/n),precision_macro=float(precision.mean()),recall_macro=float(recall.mean()),
                f1_macro=float(f1.mean()),angle_accuracy=float(matrix[a//4==b//4].sum()/n),gesture_accuracy=float(matrix[a%4==b%4].sum()/n))
