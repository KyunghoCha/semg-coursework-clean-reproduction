"""A compact Korean report from both independently verified SVM comparisons."""
import json
from pathlib import Path
import pandas as pd
from semg.paths import RUN_ROOT


def main():
    r29=RUN_ROOT/'results/svm_tuning_v29';r30=RUN_ROOT/'results/svm_regularization_v30'
    reports=[json.loads((r/'report.json').read_text()) for r in [r29,r30]]
    verified=[json.loads((r/'verification.json').read_text()) for r in [r29,r30]]
    c10=json.loads((r29/'constraint_diagnostic.json').read_text())['rows'];c1=json.loads((r30/'constraint_diagnostic.json').read_text())['rows']
    assert all(v['raw_cost_statistics_recomputed'] for v in reports)
    lines=['## 410특징 SVM의 규제와 커널 폭: v29·v30','',
      '현재 410특징·64목표 SVM에는 C100/gamma=scale 한 설정만 적용돼 있었다. 과거 240특징·직접 8분류 결과를 이 설정의 탐색으로 간주하지 않았다. v25 내부 오답 1,262개 중 875개는 정답이 상위 두 클래스에 있었고, SVM 단독은 그 오답 중 353개를 맞혔다. 이 보완성을 유지하면서 C 또는 RBF 계수만 바꾸는 제한 비교를 등록했다. XGB·LDA, 4:1 비율, fold별 SVM 비중 40/20/30%는 그대로다.','',
      'v29는 C10, gamma=1/820, gamma=2/410의 세 후보를 비교했다. 표준화 뒤 기존 gamma=scale은 수치상 1/410이었다. gamma를 1/2배·2배 바꿨으며 신호 입력 시간은 바꾸지 않았다. C10이 기존과 같은 결과를 보여, 새 바깥 점수 전에 별도 v30에 C1 하나를 등록했다. v30 등록은 04:20:47 UTC, v29 선택 고정은 04:21:55 UTC, 첫 새 바깥 결과는 04:25:11 UTC다. v29의 후보를 사후 교체하지 않았다.','',
      '|고정 SVM 설정|내부 fold0 정답|fold1 정답|fold2 정답|기준 대비 합계|',
      '|---|---:|---:|---:|---:|']
    inner29=pd.read_csv(r29/'inner.csv');inner30=pd.read_csv(r30/'inner.csv')
    base=inner29[inner29.candidate=='incumbent'].sort_values('fold').correct.to_numpy()
    for label,t,name in [('C100 / scale (기존)',inner29,'incumbent'),('C10 / scale',inner29,'regularized'),('C100 / gamma 절반',inner29,'broad'),('C100 / gamma 두 배',inner29,'narrow'),('C1 / scale',inner30,'C1')]:
        vals=t[t.candidate==name].sort_values('fold').correct.to_numpy();lines.append('|'+label+'|'+'|'.join(str(int(v)) for v in vals)+'|'+f'{int(vals.sum()-base.sum()):+d}'+'|')
    lines+=['','각 내부 분할은 256시행×19창=4,864창이며 전체 창을 유지했다. SVM 파일 크기는 기존의 1.25배 이하만 허용하고, 정답 수가 같으면 기존을 유지했다.','']
    for label,r,rep in [('v29',r29,reports[0]),('v30',r30,reports[1])]:
        old=next(v for v in rep['rows'] if v['role']=='incumbent');new=next(v for v in rep['rows'] if v['role']=='selected');names=rep['fold_candidates']
        if rep['complete']['outer_models']:
            lines.append(f"{label} 내부 선택은 {' / '.join('넓은 커널' if n=='broad' else '기존' if n=='incumbent' else n for n in names)}이며 바깥 정답은 **{new['correct']:,}/{new['windows']:,}창, {100*new['accuracy']:.4f}±{100*new['fold_sd']:.4f}%**다. v25보다 {new['correct']-old['correct']:+,}개다. 이 비교의 내부 이득과 바깥 결과를 모두 보존한다.")
        else:
            lines.append(f"{label}은 세 분할 모두 기존 유지다. 새 후보의 바깥 학습은 0회이며 **{100*new['accuracy']:.4f}%는 v25를 그대로 재사용한 값**이다. C1 후보가 그 바깥 정확도를 얻었다고 해석하지 않는다.")
        lines+=['',f'![{label}의 내부·바깥 정답 창 수 변화](results/{r.name}/comparison.png)','']
    lines += [f"C100→10의 12개 base SVM 배열이 모두 같았는지 확인한 결과는 **{'모두 정확히 일치' if all(v['fitted_svc_arrays_identical'] for v in c10) else '일부 차이 발견'}**다. 기존 최대 계수가 10보다 작아 C10이 실제 제약을 가하지 않았다. C1의 12개 base 모델 중 {sum(v['candidate_coefficients_at_C']>0 for v in c1)}개에서는 계수가 실제 상한 1에 도달했다. 따라서 C10의 무변화와 C1의 성능 결과를 같은 종류의 규제 실험으로 뭉뚱그리지 않는다.", '',
      '|측정 대상|학습 범위|전체 모델 MB|구성 모델의 관측 학습 합(초)|특징+전체 확률 중앙값(ms)|평균(ms)|p95(ms)|',
      '|---|---|---:|---:|---:|---:|---:|']
    for label,rep in zip(['v29','v30'],reports):
        for row in rep['cost']:
            lines.append(f"|{label} {row['candidate']}|{row['stage']} fold{row['fold']}|{row['model_bytes']/1e6:.4f}|{row['original_component_training_seconds_sum']:.2f}|{row['median_ms']:.4f}|{row['mean_ms']:.4f}|{row['p95_ms']:.4f}|")
    fits=sum(v['complete']['calibrated_models'] for v in reports);bases=sum(v['complete']['base_svc_fits'] for v in reports)
    lines+=['',
      '각 비교 안에서 같은 첫 20평가 시행의 380창·19예열·CPU batch1을 사용했다. 특징 계산과 모든 모델의 확률 계산을 포함하며 신호 취득·CSV 읽기·모델 적재는 제외한다. 다른 학습·복원 검증과 겹치지 않게 순차 측정했다. 내부 모델은 768시행, 바깥 모델은 1,024시행으로 학습하므로 서로 압축률로 직접 비교하지 않는다. v25의 라벨 전용 호출 생략 시간 6.60 ms와도 다른 경로다. wrapper fit 시간에는 보정 OOF 예측과 sigmoid 학습이 포함되며 base fit 합과 더해 중복 계산하지 않는다.','',
      f'새 확률모델은 총 **{fits}개**, 64분류 base SVC fit는 **{bases}개**다. 모든 모델의 4개 base 체크포인트, 실제 C/gamma, 클래스 지원, 수렴, 학습 전용 scaler, 시행 단위 보정 OOF, 64개 sigmoid 계수, 최종 확률·라벨·metric을 독립 복원했다. 선택·파일 크기 제한·분할 ID와 전 29,184창의 CPU 결합 라벨도 확인했다. 학습 시행의 19창은 보정 fold를 넘지 않는다. 원래 시험 384시행의 새 학습·평가·선택은 없다. 공유 캐시 파일에는 시험 행도 들어 있으므로 파일 수준에서 전혀 읽지 않았다고 주장하지 않는다.','',
      '검사한 범위는 C=1/10/100에서 gamma=scale, 그리고 C100에서 gamma=기준의 1/2배·2배다. C와 gamma의 모든 조합·연속 범위를 소진한 것은 아니다. 현재 결과는 넓은 커널의 미세한 내부 이득이 바깥에서 유지되지 않았고, 더 좁은 커널과 C1은 내부에서 유리하지 않았다는 범위로 해석한다. 후속 범위를 무작정 넓히는 것보다 새 표현이나 다른 학습기의 보완 오류에 대한 내부 근거를 먼저 요구한다.','',
      '현재 200 ms·100 ms 이동·전 19창·개발 1,536시행·64기록을 유지했고 과거/미래 창·시행 투표·기록 ID 추론 입력은 없다. 반복 개발 선택과 이전 바깥 점수 노출은 남아 있다. 64학습 목표와 64기록은 일대일이므로 기록 공유 성능을 새 기록의 동작 일반화로 제시하지 않는다. 논문 전문·정확한 원필터·독립 세션은 미확보이며, 논문 정확재현이나 독립 외부 시험 성공으로 주장하지 않는다. 기존 v23·v25 및 공개 묶음은 보존하고 외부 게시·업로드는 하지 않았다.','',
      '[v29 사전 명세](experiments/svm_tuning_v29/PLAN.md), [v30 사전 명세](experiments/svm_regularization_v30/PLAN.md), [v29 검증](results/svm_tuning_v29/verification.json), [v30 검증](results/svm_regularization_v30/verification.json), [v29 비용](results/svm_tuning_v29/costs.json), [v30 비용](results/svm_regularization_v30/costs.json)에 원시 집계와 검증 범위를 남겼다.']
    (RUN_ROOT/'private/svm_tuning_v29/README_SECTION.md').write_text('\n'.join(lines)+'\n')
    print('Verified reports:',fits,'calibrated models;',bases,'base SVC fits')


if __name__=='__main__':main()
