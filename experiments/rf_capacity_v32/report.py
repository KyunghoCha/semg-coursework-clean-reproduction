"""Render verified RF results without changing experiment settings or scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/rf_capacity_v32'
verification=json.loads((R/'verification.json').read_text())
fits=pd.read_csv(R/'fits.csv');inner=pd.read_csv(R/'inner.csv');pooled=pd.read_csv(R/'pooled.csv')
selection=json.loads((R/'selection.json').read_text())
assert verification['models_restored']==len(fits)
rows=[]
for name in ['direct8','latent64']:
    t=fits[(fits.stage=='inner')&(fits.name==name)].sort_values('fold')
    b=inner[inner.name=='incumbent'].sort_values('fold')
    assert len(t)==len(b)==3
    delta=t.correct.to_numpy()-b.correct.to_numpy()
    rows.append(dict(name=name,solo_mean=float(t.solo_accuracy.mean()),solo_sd=float(t.solo_accuracy.std(ddof=0)),
        mixture_mean=float(t.accuracy.mean()),inner_correct_delta=delta.tolist(),
        bytes_min=int(t.model_bytes.min()),bytes_max=int(t.model_bytes.max()),
        fit_seconds_total=float(t.fit_seconds.sum()),fit_seconds_mean=float(t.fit_seconds.mean()),
        training_accuracy_min=float(t.training_accuracy.min()),training_accuracy_max=float(t.training_accuracy.max())))
summary=dict(models=rows,fold_candidates=selection['fold_candidates'],pooled=pooled.to_dict('records'),
    test_evaluations=0,outer_candidate_fits=int((fits.stage=='outer').sum()),input_ms=200,windows_per_trial=19,
    calibration='RF bagged probabilities; fixed 10% mixture, no additional fitted calibration',
    latency=('Skipped: no RF passed the inner adoption rule' if not (fits.stage=='outer').any() else 'Representative adopted-candidate cost measurement pending'))
path=R/'summary.json'
if path.exists(): assert json.loads(path.read_text())==summary
else:path.write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
colors=['#326A9A','#C47332'];x=np.arange(3)
for j,row in enumerate(rows):
    name='Direct 8 classes' if row['name']=='direct8' else '64 targets -> 8 classes'
    bars=ax[0].bar(x+(j-.5)*.34,row['inner_correct_delta'],.34,label=name,color=colors[j])
    ax[0].bar_label(bars,fmt='%+g',padding=3,fontsize=9)
ax[0].axhline(0,color='black',linewidth=.7);ax[0].axhline(10,color='gray',linestyle='--',label='Required gain: +10')
ax[0].set(xticks=x,xticklabels=['Fold 0','Fold 1','Fold 2'],ylabel='Change in correct windows (of 4,864)',title='Fixed 10% RF mixture: inner results')
ax[0].margins(y=.35);ax[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=1)
for j,row in enumerate(rows):
    bars=ax[1].bar(j,row['solo_mean']*100,color=colors[j],width=.6)
    ax[1].bar_label(bars,labels=[f"{row['solo_mean']*100:.2f}%"],padding=4)
ax[1].set(xticks=[0,1],xticklabels=['Direct 8','64 -> 8'],ylim=(0,100),ylabel='Mean inner window accuracy (%)',title='RF alone: same 410 features / 200 ms')
fig.savefig(R/'inner_comparison.png',dpi=160);plt.close(fig)
lines=['## v32: 같은 410개 특징에서 RF 용량과 학습 목표 비교','',
'논문 비교표에서 강의 RF 100트리·25TD 기준선의 임의 축소는 확인되지 않았다. 별개로 현재 410개 특징에서 RF 자체를 충분한 잎 용량으로 비교한 기록이 없었다. v13은 905개 특징·직접 8목표였고 v28은 잎 제한 ExtraTrees였다. 따라서 RF 200트리·max_features=.5·깊이/잎 제한 없음 한 설정을 직접 8/64목표에서 비교했다. 같은 200ms·100ms 이동·19창·기존 시행 분할·전체 개발 1,536시행을 유지했다.','',
'|목표|RF 단독 내부 평균±SD|기존 결합에 10% 추가한 정답 변화(3 fold)|모델 크기 범위|평균 학습시간|','|---|---:|---|---:|---:|']
for r in rows:
    delta='/'.join(f'{d:+d}' for d in r['inner_correct_delta'])
    lines.append(f"|{r['name']}|{100*r['solo_mean']:.4f}±{100*r['solo_sd']:.4f}%|{delta}창|{r['bytes_min']/1e6:.2f}–{r['bytes_max']/1e6:.2f} MB|{r['fit_seconds_mean']:.2f}초|")
lines += ['',f"내부 후보 6개를 학습했고 선택은 `{selection['fold_candidates']}`였다. 내부 정답 최소 10개 증가·전체 크기 기존 3배 이내를 요구했다. 새 바깥 RF는 {summary['outer_candidate_fits']}개다. 바깥 결과의 기존 모델 재사용과 새 후보 평가를 구분한다.", '',
'![RF 용량 대조의 내부 성능과 정답 변화](results/rf_capacity_v32/inner_comparison.png)','',
'그림. 같은 내부 평가 4,864창의 정답 변화와 RF 단독 정확도다. 상관된 창을 독립 표본으로 가정한 신뢰구간은 계산하지 않았다. 64목표가 기록과 일대일이라는 한계가 유지되며, 이 RF 대조를 논문 96.72%의 재현으로 부르지 않는다.','',
f"{verification['models_restored']}개 저장모델 복원, 각 200트리·전체 학습 창 제공·bootstrap 표본 수·학습 전용 scaler·확률 열·독립 선택 규칙·전 29,184바깥 창 집계를 확인했다. 기존 시험 384시행은 재평가하지 않았다. 초기 검사 수집은 파일명 중복으로 중단됐고 새 검사명만 고유하게 바꾼 뒤 75개 검사를 통과했다. 열세 내부 후보의 긴 지연 측정은 생략했다. [사전 계획](experiments/rf_capacity_v32/PLAN.md), [집계](results/rf_capacity_v32/summary.json), [검증](results/rf_capacity_v32/verification.json)."]
(ROOT/'private/rf_capacity_v32/README_SECTION.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
