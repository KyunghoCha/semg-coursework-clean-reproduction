# 복구한 표현에서 무작위 숲의 분산 감소 대조

v8과 v9 좁은 표현의 첫 fold 모델은 학습 19,456창을 8클래스와 64개 목표 모두 100% 맞혔지만, 같은 fold 평가 정확도는 89.7512/90.0493%였다. 학습 log loss는 .00440/.00471, 평가 log loss는 .30475/.29731이었다. 학습 용량 부족만으로 남은 오류를 설명하기 어렵다. 이 진단은 학습 자료 재평가이며 일반화 점수가 아니다.

다음 가설은 여러 무작위 트리를 평균하는 구조가, 세부 주파수·채널 관계를 복구한 현재 입력에서 오차를 줄이는가이다. 초기 240특징 비교에서 ExtraTrees400의 내부 평균은 77.5937%, 같은 400회 XGB는 77.1655%였으나 최신 905특징에 숲 모델을 아직 적용하지 않았다. 알고리즘 차이에 대한 가설이며 효과를 미리 가정하지 않는다. 새 특징·신호 구간·시행 선택은 추가하지 않는다.

## 고정한 모델과 선택 범위

새 후보는 두 개뿐이다. 둘 다 직접 8클래스, 기존 v8의 905특징, StandardScaler를 실제 학습 부분에만 fit, random_state=42, n_jobs=6이다.

1. RF100_fine: RandomForestClassifier, n_estimators=100, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=None, ccp_alpha=0.0. 그 밖에는 설치된 scikit-learn1.7.2 기본값을 사용하고 전체 get_params를 저장한다. 강의 RF100의 트리 수를 그대로 유지한 표현 확장이다.
2. ET400_fine: ExtraTreesClassifier, n_estimators=400, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=.5, bootstrap=False, class_weight=None, ccp_alpha=0.0. 초기 ET의 모델 설정을 유지한 표현 확장이다. 나머지 전체 설정도 저장한다.

이미 등록한 v10 내부 학습 768시행·선택 256시행 3개 분할에 두 모델을 학습한다. 새 내부 학습은 6회다. 같은 내부 자료의 기존 fine64와 narrow64 결과도 후보로 포함하며 다시 학습하지 않는다. 각 fold의 내부 정답 창 수가 가장 큰 후보를 선택한다. 정확히 동점이면 실제 저장 모델·scaler 바이트가 작은 후보, 그다음 사전 순서 fine64/narrow64/RF100_fine/ET400_fine를 사용한다. 두 숲 모델의 OOB 점수는 선택에 쓰지 않는다.

선택된 모델을 해당 바깥 학습 1,024시행에서 다시 학습하고 바깥 512시행의 모든 19창을 평가한다. 선택이 기존 XGB이면 v10 또는 v12에서 같은 분할·설정으로 준비한 모델을 해시 확인 뒤 재사용한다. 따라서 새 바깥 학습은 0~3회, 새 학습 전체는 6~9회다. 고정 fine64 바깥 결과를 대조로 보존한다. 최종 개발 후보는 내부 세 분할의 정답 수 합, 동점이면 내부 저장 바이트 합, 그다음 사전 순서로 정한다. 바깥 결과로 후보를 다시 선택하지 않는다.

## 평가와 비용

200ms·100ms 이동·19창·5채널·개발 1,536시행과 원래 crop을 유지한다. 기존 384시행 시험은 학습·선택·평가에 쓰지 않는다. 숲은 설치된 scikit-learn CPU 모델이며 같은 사용자 PC에서 실행한다. GPU를 지원하는 XGB는 기존 GPU 학습 결과를 재사용한다. 다른 학습이나 추론 벤치마크와 동시에 실행하지 않는다.

현재 창 정확도·macro F1·각도/동작 정확도·log loss를 계산한다. ±는 fold 표준편차(ddof=0)다. 모든 실제 모델의 트리·노드·잎·저장 바이트와 학습 시간을 기록한다. XGB의 클래스별 scalar 트리와 숲의 다중 클래스 트리는 구조가 다르므로 트리 수만으로 동일 용량이라고 주장하지 않는다. 유망한 새 모델은 안정된 시점에 같은 CPU 6스레드의 단일 창 중앙값/p95와 처리량을 측정한다.

저장 모델 복원, classes_와 확률 열 순서, scaler fit 범위, split ID, 수동 지표와 원본 재사용 파일의 해시를 확인한다. 실패한 후보도 전체 내부 결과와 비용을 남긴다. 기존 v10·v11·v12를 변경하지 않고 완료한 뒤 실행한다. 이 계획은 v10의 첫 바깥 narrow64 89.5662%를 본 시점에 작성했으며 숲 결과는 아직 없다. 과거 개발 노출·반복 선택·기록 공유의 편향이 남는 조건부 개발 비교다. 기록 조건 의존을 해결했다고 주장하지 않는다.

근거: 설치 버전에 맞춘 [RandomForest 문서](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [ExtraTrees 문서](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html). 무작위화·평균의 설명을 참고했으며 이 자료에서의 효과는 위 고정 대조로 판단한다.
