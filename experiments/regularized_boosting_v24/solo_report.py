"""Auxiliary standalone scores of XGB checkpoints selected for the mixture, no selection."""
import json
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from data import P,R,baseline,digest


def main():
    assert (R/'verification.json').exists()
    if (R/'solo_outer_summary.json').exists():raise FileExistsError('Preserve auxiliary scores')
    names=json.loads((R/'selection.json').read_text())['fold_candidates'];rows=[];truths=[];preds=[];hashes={}
    for fold,name in enumerate(names):
        path=baseline('outer',fold) if name=='incumbent' else P/f'outer_{name}_{fold}.npz';sha=digest(path);z=np.load(path);pred=np.stack([z['probability'][:,c::8].sum(1) for c in range(8)],axis=1).argmax(1);t=z['truth'];rows.append(dict(fold=fold,candidate=name,windows=len(t),correct=int((t==pred).sum()),**manual_metrics(t,pred)));truths.append(t);preds.append(pred);hashes[str(path.relative_to(RUN_ROOT))]=sha;assert digest(path)==sha
    t=np.concatenate(truths);p=np.concatenate(preds);out=dict(scope='auxiliary standalone XGB; candidates were selected for fixed three-model mixture',windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std([r['accuracy'] for r in rows])),**manual_metrics(t,p),source_hashes=hashes,new_model_fits=0,selection_changed=False,input_ms=200)
    pd.DataFrame(rows).to_csv(R/'solo_outer.csv',index=False);(R/'solo_outer_summary.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
