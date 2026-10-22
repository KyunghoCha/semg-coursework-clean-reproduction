"""Post-result training accuracy from preserved models; no fitting or selection."""
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from qda_data import inputs,P,R
from gaussian_boundary import to_eight


def main():
    if (R/'training_diagnostic.csv').exists():raise FileExistsError('Preserve diagnostic')
    x,y,target,schedule=inputs();rows=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_train']);o=joblib.load(P/f'inner_qda_{k}.joblib');q=o['model'].predict_proba(o['scaler'].transform(x[ids].reshape(-1,410)))
        rows.append(dict(fold=k,train_windows=len(q),train_8class_accuracy=float((to_eight(q).argmax(1)==np.repeat(y[ids],19)).mean()),train_64class_accuracy=float((q.argmax(1)==np.repeat(target[ids],19)).mean())))
    pd.DataFrame(rows).to_csv(R/'training_diagnostic.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
