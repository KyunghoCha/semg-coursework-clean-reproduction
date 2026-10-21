"""Generate only group-level tables and plots in the public results directory."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from semg.data import CLASSES, CHANNELS, marker_indices
from semg.paths import RUN_ROOT

ROOT=RUN_ROOT
P=ROOT/'private';R=ROOT/'results';F=R/'figures'
NAMES=['ECR','ED','FDS','Biceps','Triceps']


def save(fig,name):
    fig.tight_layout();fig.savefig(F/name,dpi=160);plt.close(fig)


def signals(raw):
    profiles=[]
    for path in sorted(Path(raw).glob('*.csv')):
        d=pd.read_csv(path);marks=marker_indices(d);x=d[CHANNELS].to_numpy()
        v=np.stack([x[i:i+4900] for i in marks])
        profiles.append(np.sqrt(np.mean(v.reshape(30,49,100,5)**2,axis=2)))
    profiles=np.stack(profiles)
    np.save(P/'cache/rms_profiles.npy',profiles)
    fig,ax=plt.subplots(figsize=(10,4))
    summary=[]
    for c,name in enumerate(NAMES):
        rr=np.median(profiles[:,:,:,c],axis=1)
        rr/=np.maximum(rr.max(axis=1,keepdims=True),1e-12)
        group=np.median(rr,axis=0)
        ax.plot(np.arange(49)*.1+.05,group,label=name)
        summary.extend(dict(channel=name,seconds_after_mark=.05+.1*j,
                            median_normalized_rms=float(value),recordings=64,subjects=8)
                       for j,value in enumerate(group))
    pd.DataFrame(summary).to_csv(R/'marker_profile_summary.csv',index=False)
    ax.legend(ncol=5,fontsize=8);ax.set_ylabel('Group median normalized RMS')
    ax.set_xlabel('Seconds after mark')
    ax.axvspan(1.5,3.5,alpha=.1,color='red')
    ax.axvline(2,color='gray',ls=':');ax.axvline(3,color='gray',ls=':')
    save(fig,'marker_alignment.png')


def channel_analysis():
    seg=np.load(P/'cache/segments.npy',allow_pickle=False)
    meta=pd.read_csv(P/'trials.csv')
    rms=np.sqrt((seg**2).mean(axis=1))
    summary=[]
    for c,name in enumerate(NAMES):
        data=meta[['subject','angle','gesture']].copy();data['rms']=rms[:,c]
        subject=data.groupby(['subject','gesture','angle']).rms.median().unstack('angle')
        for g,d in data.groupby('gesture'):
            z=subject.xs(g,level='gesture');ratios=z[60]/z[30]
            summary.append(dict(gesture=g,channel=name,median_ratio=float(ratios.median()),
                                subjects_increase=int((ratios>1).sum()),subjects=8))
    table=pd.DataFrame(summary);table.to_csv(R/'channel_angle_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    gestures=['extor','fist','open','pro']
    ratios=table.pivot(index='gesture',columns='channel',values='median_ratio').loc[gestures,NAMES]
    limit=max(1.,float(np.abs(np.log2(ratios.to_numpy())).max()))
    im=ax.imshow(np.log2(ratios),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i,g in enumerate(gestures):
        for j,ch in enumerate(NAMES):
            ax.text(j,i,f'{ratios.loc[g,ch]:.3f}',ha='center',va='center',fontsize=11)
    ax.set_xticks(range(5),NAMES);ax.set_yticks(range(4),gestures)
    ax.set_xlabel('Channel');ax.set_ylabel('Original gesture code')
    fig.colorbar(im,ax=ax,label='log2(group median RMS ratio: 60 / 30 deg)')
    save(fig,'channel_angle_medians.png')


def performance():
    df=pd.read_csv(R/'baseline.csv');cv=df[df.phase=='baseline_cv']
    summary=cv.groupby('model',sort=False).agg(mean_accuracy=('accuracy','mean'),
                std_accuracy=('accuracy',lambda a:np.std(a,ddof=0)),mean_f1=('f1_macro','mean'))
    summary.to_csv(R/'baseline_cv_summary.csv')
    held=df[df.phase=='baseline_holdout'].set_index('model')
    fig,ax=plt.subplots(figsize=(8,4))
    order=['SVM','RF','XGB'];x=np.arange(3)
    ax.bar(x-.18,held.loc[order].accuracy*100,width=.36,label='Hold-out')
    ax.bar(x+.18,summary.loc[order].mean_accuracy*100,width=.36,
           yerr=summary.loc[order].std_accuracy*100,label='5-fold mean (SD)')
    ax.set_xticks(x,order);ax.set_ylim(0,100);ax.set_ylabel('Window accuracy (%)');ax.legend()
    save(fig,'baseline_accuracy.png')
    errors=[];position=[]
    for name in order:
        d=np.load(P/f'baseline_holdout/{name}_0_predictions.npz',allow_pickle=False)
        cm=confusion_matrix(d['truth'],d['prediction'],labels=range(8))
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(R/f'confusion_{name}.csv')
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap='Blues',vmin=0)
        ax.set_xticks(range(8),CLASSES,rotation=40,ha='right');ax.set_yticks(range(8),CLASSES)
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
        for i in range(8):
            for j in range(8):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                        color='white' if cm[i,j]>cm.max()*.55 else 'black')
        fig.colorbar(im,ax=ax,label='Windows');save(fig,f'confusion_{name}.png')
        for phase in ['baseline_holdout','baseline_cv']:
            files=sorted((P/phase).glob(f'{name}_*_predictions.npz'))
            truths=[];predictions=[];trials=[];starts=[]
            for file in files:
                d=np.load(file,allow_pickle=False)
                truths.extend(d['truth']);predictions.extend(d['prediction']);trials.extend(d['trial'])
                starts.extend(np.tile(np.arange(19),len(d['truth'])//19))
            t=np.array(truths);p=np.array(predictions);ids=np.array(trials);starts=np.array(starts)
            ae=t//4!=p//4;ge=t%4!=p%4
            errors.append(dict(model=name,phase=phase,total=len(t),correct=int((t==p).sum()),
                               angle_only=int((ae&~ge).sum()),gesture_only=int((~ae&ge).sum()),
                               both=int((ae&ge).sum())))
            if phase=='baseline_cv':
                for j in range(19):
                    take=starts==j
                    position.append(dict(model=name,start_seconds=1.5+j*.1,
                                         accuracy=float((t[take]==p[take]).mean()),windows=int(take.sum())))
    pd.DataFrame(errors).to_csv(R/'error_categories.csv',index=False)
    position=pd.DataFrame(position);position.to_csv(R/'window_position_cv.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    for name,d in position.groupby('model',sort=False):
        ax.plot(d.start_seconds,d.accuracy*100,marker='o',ms=3,label=name)
    ax.set_ylim(0,100);ax.set_ylabel('OOF window accuracy (%)');ax.set_xlabel('Window start after mark (s)')
    ax.legend();save(fig,'window_position.png')
    if (R/'development.csv').exists():
        dev=pd.read_csv(R/'development.csv')
        ds=dev.groupby('model',sort=False).agg(mean_accuracy=('accuracy','mean'),
                std_accuracy=('accuracy',lambda a:np.std(a,ddof=0)),mean_f1=('f1_macro','mean'),
                mean_angle_accuracy=('angle_accuracy','mean'),mean_gesture_accuracy=('gesture_accuracy','mean'))
        ds.to_csv(R/'development_summary.csv')
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(ds.index,ds.mean_accuracy*100,xerr=ds.std_accuracy*100)
        ax.set_xlim(0,100);ax.set_xlabel('Development window accuracy (%)');ax.invert_yaxis()
        save(fig,'development_accuracy.png')


def selected_analysis():
    if not (R/'selected_holdout.csv').exists():return
    name=pd.read_csv(R/'selected_holdout.csv').model.iloc[0]
    d=np.load(P/f'selected_holdout/{name}_0_predictions.npz',allow_pickle=False)
    t,p=d['truth'],d['prediction']
    cm=confusion_matrix(t,p,labels=range(8))
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(R/'confusion_selected.csv')
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap='Blues',vmin=0)
    ax.set_xticks(range(8),CLASSES,rotation=40,ha='right');ax.set_yticks(range(8),CLASSES)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    for i in range(8):
        for j in range(8):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]>cm.max()*.55 else 'black')
    fig.colorbar(im,ax=ax,label='Windows');save(fig,'confusion_selected.png')
    ae=t//4!=p//4;ge=t%4!=p%4
    pd.DataFrame([dict(model=name,total=len(t),correct=int((t==p).sum()),
                       angle_only=int((ae&~ge).sum()),gesture_only=int((~ae&ge).sum()),
                       both=int((ae&ge).sum()))]).to_csv(R/'selected_errors.csv',index=False)
    # Post-selection diagnosis only: same first 18 start positions, different durations.
    # Training still used 19 windows for 200 ms versus 18 for 300 ms.
    # This is not a training-sample-controlled comparison; no refitting occurs.
    if name!='XGB_win300':return
    rows=[]
    for phase,count in [('development',5),('holdout',1)]:
        for k in range(count):
            for model in ['SVM','RF','XGB','XGB_win300']:
                source=phase if phase=='development' else (
                    'selected_holdout' if model==name else 'baseline_holdout')
                z=np.load(P/f'{source}/{model}_{k}_predictions.npz',allow_pickle=False)
                n=18 if model==name else 19
                truth=z['truth'].reshape(-1,n)[:,:18]
                prediction=z['prediction'].reshape(-1,n)[:,:18]
                rows.append(dict(phase=phase,fold=k,model=model,window_ms=300 if n==18 else 200,
                                 start_positions=18,windows=truth.size,
                                 accuracy=float((truth==prediction).mean())))
    pd.DataFrame(rows).to_csv(R/'matched_start_diagnostic.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw');args=parser.parse_args()
    F.mkdir(exist_ok=True)
    if args.raw: signals(args.raw)
    channel_analysis()
    if (R/'baseline_complete.json').exists():
        performance()
        selected_analysis()
