"""Report independently verified single-model performance and cost status."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/standalone_xgb_v33';P=ROOT/'private/standalone_xgb_v33'
verify=json.loads((R/'verification.json').read_text());sel=json.loads((R/'selection.json').read_text());fits=pd.read_csv(R/'fits.csv');inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');pool=pd.read_csv(R/'pooled.csv')
assert verify['models_restored']==len(fits)
old=inner[inner.model=='depth4_800'].sort_values('fold');new=inner[inner.model=='depth3_1600'].sort_values('fold')
delta=(new.correct.to_numpy()-old.correct.to_numpy()).tolist()
rows=pool.to_dict('records');report=dict(input_ms=200,windows_per_trial=19,features=410,models_per_prediction=1,output_blending=False,
    use_new_model=sel['use_new_model'],inner_correct_delta=delta,outer_results=rows,new_fits=len(fits),
    historical_test_evaluations=0,external_independent_test=False,
    standalone_inventory='standalone_inner_summary.csv',cost_status='costs.json' if (R/'costs.json').exists() else 'Only fit times and sizes; inference measurement skipped or pending')
dest=R/'summary.json'
if dest.exists():assert json.loads(dest.read_text())==report
else:dest.write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(1,2,figsize=(10.7,4.5),layout='constrained');x=np.arange(3)
for offset,table,label,color in [(-.18,old,'Depth 4 / 800','#386B9A'),(.18,new,'Depth 3 / 1,600','#C37335')]:
    bars=ax[0].bar(x+offset,table.accuracy.to_numpy()*100,.36,label=label,color=color);ax[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax[0].set(xticks=x,xticklabels=['Fold 0','Fold 1','Fold 2'],ylim=(0,100),ylabel='Single-window accuracy (%)',title='Inner selection: one XGB classifier')
ax[0].legend(loc='lower center',ncol=1)
for j,row in enumerate(rows):
    bars=ax[1].bar(j,row['accuracy']*100,width=.6,color=['#386B9A','#45947A'][j]);ax[1].bar_label(bars,labels=[f"{row['accuracy']*100:.4f}%"],padding=4)
ax[1].set(xticks=[0,1],xticklabels=['Fixed depth 4 / 800','Inner-selected XGB'],ylim=(0,100),ylabel='Single-window accuracy (%)',title='Outer development assessment / 29,184 windows')
fig.savefig(R/'single_model_comparison.png',dpi=160);plt.close(fig)
lines=['## v33: 결합 없는 단독 XGBoost의 고정 대조','',
'같은 내부 분할에서 XGB800의 단독 정답이 XGB400보다 세 번 모두 높았다. 더 얕은 트리에 학습 단계를 늘리는 한 후보(깊이3·1,600회)를 GPU에서 검증했다. 사용자의 새 조건에 따라 한 번의 예측에는 XGB 한 모델만 쓰며 LDA·SVM·RF 출력이나 과거 창을 결합하지 않는다. 기존 200ms·100ms 이동·19창·개발 1,536시행·410특징·시행 분할을 유지했다.','',
'|내부 fold|기존 단독 XGB800|새 단독 XGB1600|정답 변화|고정 선택|','|---|---:|---:|---:|---|']
for k,(a,b,d,use) in enumerate(zip(old.to_dict('records'),new.to_dict('records'),delta,sel['use_new_model'])):
    lines.append(f"|{k}|{a['accuracy']*100:.4f}%|{b['accuracy']*100:.4f}%|{d:+d} / 4,864창|{'깊이3·1600' if use else '깊이4·800'}|")
ref=pool[pool.role=='reference'].iloc[0];chosen=pool[pool.role=='selected'].iloc[0]
lines += ['',f"내부에서 정답 최소10개 증가·기존 모델의2배 이하 크기를 요구하고 선택을 먼저 고정했다. 바깥 개발 결과는 기존 단독 **{ref.accuracy*100:.4f}%({int(ref.correct)}/29,184)**에서 선택 절차 **{chosen.accuracy*100:.4f}±{chosen.fold_sd*100:.4f}%({int(chosen.correct)}/29,184)**다. 순변화는{int(chosen.correct-ref.correct):+d}창이다. 이 값은 과거91.9819% 앙상블과 다른 단독 모델 조건이며, 독립 외부 시험도 아니다.", '',
'![단독 XGB의 내부 선택과 바깥 개발 결과](results/standalone_xgb_v33/single_model_comparison.png)','',
f"새{len(fits)}fit와 저장모델{verify['models_restored']}개를 복원했다. 모든 새 fit에서 CUDA·1,600회·102,400트리, 학습 전용 scaler, 같은 분할·클래스 지원·입력 해시·단일 모델 확률 해독·CPU 복원 라벨·수동 지표·기존 산출물 해시가 일치했다. 전체77개 검사도 통과했다. 기존 시험384시행은 평가하지 않았다.", '',
'새 후보의 학습 시간과 모델 크기는 [fit 기록](results/standalone_xgb_v33/fits.csv)에 있다. 정확도에 불리한 후보의 긴 비용 측정은 하지 않는다. 비용 측정을 완료한 경우 아래 표와 원시 집계에서 대표 fold·380창·현재창 특징 계산 포함 여부를 명시한다.','',
'이전 바깥 결과 노출과 반복 내부 선택은 남아 있다. 64목표와64기록의 일대일 관계도 유지되므로 새 기록 일반화로 해석하지 않는다. 논문의384개 표본 정의·구간 시작점·실제 필터와 원자료 대응은 미확인이다. [사전 계획](experiments/standalone_xgb_v33/PLAN.md), [검증](results/standalone_xgb_v33/verification.json), [집계](results/standalone_xgb_v33/summary.json).']
(P/'README_SECTION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,indent=2))
