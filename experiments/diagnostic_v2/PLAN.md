# 개발 진단 후 2차 개선 계획

이 계획은 2차 개선 모델 점수를 보기 전에 고정한다. 이전 1차 기준·개선 결과와 공개용 export를 보존한다. 지금부터는 clean 프로젝트의 별도 브랜치와 `diagnostic_v2` 하위 결과만 추가한다. 고정 시험 384시행은 새 후보 선택과 진단에서 제외하며, 이미 전체 CV에 노출된 사실을 유지한다.

## 확인한 진단

개발 1536시행의 기존 OOF 예측과 신호만 사용했다. ZC=0 비율은 삼두근 60.3%, 이두근 48.5%이고 전체 창의 13.3%에서 5채널 ZC가 모두 0이다. 0.01 임계값과 비슷하거나 작은 RMS가 많아 저진폭 파형의 부호 정보를 추가로 확인한다. 원문 baseline의 임계값은 수정하지 않는다.

XGB의 위치별 개발 정확도는 1.7초 시작에서 72.85%, 2.4초에서 53.84%, 3.3초에서 65.23%였다. 조용한 후반부보다 에너지가 감소하는 전이 구간에서 동작 오류가 커졌다. 휴식 대비 진폭비 1–3인 창은 약55%이며 1 미만은64.18%, 10 이상은77.26%다. 어려운 창을 제외하지 않고 19창을 유지한다.

현재 scaler는 학습 전체의 특징별 선형 표준화로 절대 진폭의 순서를 보존한다. 개인·창마다 진폭을 정규화한 것이 아니다. SVM의 저진폭 해상도와 긴 분포 꼬리 영향은 train-only QuantileTransformer로 별도 비교한다.

## 고정 입력과 특징 정의

모든 주비교는 같은 mark+1.5–3.5초, 200ms/100ms, 19창/시행, 5채널, 8클래스 창 정확도다. 전체 원본 1920시행의 특징을 통계 학습 없이 계산하되 모델 선택·진단에는 기존 hold-out 학습1536시행만 사용한다. 채널·창·피험자·조건을 제외하지 않는다. 다수결을 주점수로 사용하지 않는다.

- base: 강의25특징 그대로.
- zc0: base의 ZC5개만 임계값0 부호 교차수로 교체. 0표본은 교차로 세지 않음.
- time: base에 ZC0, ZC 임계값0.1×해당 채널 RMS, 평균·표준편차, |x|의10/25/50/75/90백분위, 표준화3·4차 적률(4차에서3제거), 차분RMS, 창 내부40ms×5구간 RMS를 추가. 표본 밖 정보를 사용하지 않는다.
- relation: base에 채널 RMS의 L2 상대비5개, 로그 RMS 쌍차10개, 중심화 신호 Pearson 상관10개, |x| 중심화 상관10개를 추가. base의 절대 진폭은 유지한다.
- spectrum: base에 200표본 Hann periodogram(fs1000,detrend=constant)의8대역[0–20,20–50,50–100,100–150,150–250,250–350,350–450,450–500] log10 절대전력 및 상대전력(각40개), MNF/MDF/정규화 spectral entropy(각5개)를 추가. 오른쪽 끝 제외, 마지막은500Hz 포함. 절대전력은PSD 합×주파수간격5Hz, 상대전력은 전체PSD 합으로 나눈다.
- rich: base+time추가분+relation추가분+spectrum추가분. epsilon=1e-12, 0분산 상관은0. 기존 사전 필터 입력에 새 필터를 중복 적용하지 않는다.

## 첫 모델 비교와 선택 범위

후보는 아래10개로 고정한다. 추가 후보가 필요하면 기존 결과를 보존하고 별도 다음 계획을 먼저 기록한다.

1. XGB_base: 25D, 강의400회/depth6/lr.05/subsample.9/colsample.8/lambda1/hist/CUDA/seed42.
2. XGB_zc0: ZC 임계값만 교체.
3. XGB_time: 창 내부 분포·시간 변화 추가.
4. XGB_relation: 채널 관계 추가.
5. XGB_spectrum: 대역별 분포 추가.
6. XGB_rich: 세 특징 묶음 결합.
7. XGB_rich800: rich, boosting회수만800으로 변경.
8. ET_rich: rich, ExtraTrees400, max_features=.5, min_samples_leaf=1, bootstrap=False, seed42,n_jobs=6. 구조 비교이며 XGB와 동일 알고리즘 비교가 아니다.
9. SVM_standard: base, RBF C10 gamma=scale, StandardScaler.
10. SVM_quantile: base, RBF C10 gamma=scale, QuantileTransformer(output_distribution=normal,n_quantiles=min(1000,학습창수),subsample=None,random_state42). scaler만 변경.

XGB/ET도 해당 학습 창에만 StandardScaler를 fit한다. 전처리 통계는 각 inner/outer 학습 집합에서 새로 계산한다. GPU 모델은 실제 CUDA를 검사한다. 조기종료는 쓰지 않는다.

## 평가와 다음 판단

개발1536시행 안에 새로운 층화 outer3-fold(seed20261020), 각 outer 학습 안의 inner3-fold(seed20261021+k)를 고정한다. 모든 후보를 동일 inner folds에서 비교한다. 평균 창 정확도 최대, 동률은 위 후보 순서로 선택한다. 각 outer 평가에는 선택 모델과 XGB_base 대조만 학습해 비교한다. Outer의 평가 시행은 해당 inner 선택과 학습에서 배제한다. 평균±SD와 모든 outer OOF 창을 함께 보고한다.

동일 개발 자료는 앞선 실험과 진단에 이미 사용됐다. 중첩 CV는 이번 선택 과정 안의 재사용을 줄이지만 외부 독립 증거를 만들지는 않는다. Outer3개는 작아 불확실성을 함께 보고하며, outer 성능을 본 뒤 같은 fold에 새 후보를 맞추지 않는다. 기존 고정 시험은 이 단계에서 추가로 평가하지 않는다.

각 outer의 inner 후보별 전체 결과는 요인분리 근거다. 상승이 없는 특징은 그 실패를 기록한다. Inner 평균에서 rich가 base보다2%p 이상 높으나 800회 이득이1%p 미만이면 추가 트리 확대를 우선하지 않는다. 스펙트럼이나 채널관계가 주효하면 이후 해당묶음의 기여를 더 분리한다. ZC0가 유효하면 원래 임계값의 정보 손실을 확인하되 원본baseline 결과를 대체하지 않는다. SVM분포변환이 유효하면 scale 문제가 아니라 비선형 거리 배치의 영향을 근거로 논의한다.

일관된 창 분류 성능이 충분히 오르지 않으면 전이구간의 정보와 구조를 다시 분석한다. 2초 문맥이나 시간 모델을 사용하는 대안은 200ms 단독 입력과 구분한 별도 변경으로만 평가하며, 창 삭제나 다수결로 목표 달성을 선언하지 않는다.

근거: [중첩 CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), [periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html). 실행 API는 설치된scikit-learn1.7.2/SciPy1.16.3/XGBoost3.0.5를 사용한다.
