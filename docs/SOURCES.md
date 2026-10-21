# 근거와 접근 범위

강의의 실험 조건은 [강의 자료와 출처 범위](LECTURE_REVIEW.md)과 [재현 명세](PROTOCOL.md)에서 추적한다. 아래 외부 자료는 API와 검증 방법을 확인하는 데 사용했다. 기존 프로젝트의 코드·결과·문서는 구현 근거로 읽거나 가져오지 않았다.

| 자료 | 읽은 범위와 적용 |
|---|---|
| [논문 공개 페이지](https://link.springer.com/article/10.1007/s13534-026-00607-7) | 제목·초록·Data availability·구독 안내. 8명/5채널/8클래스와 RF96.72±.58%는 초록에서 확인. 방법 전문은 구독 콘텐츠여서 미확보 |
| [scikit-learn 데이터 누수](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | fit/transform, 학습 부분에서만 전처리 통계 계산. 각 split의 scaler fit 검증에 반영 |
| [scikit-learn 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html) | 테스트로 모델 선택할 때의 위험, 그룹 단위 분할. 시행 내부 창을 같은 집합에 유지 |
| [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html) | device=cuda와 hist 사용법. 최신 문서의 CUDA 최소 버전을 설치된3.0.5에 그대로 대입하지 않고 로컬 build_info와 실제 GPU 학습으로 확인 |
| [SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html) | 단측 PSD, 샘플링, 창, detrend, axis 정의. MNF/MDF 확장 설정을 명시적으로 고정 |


## 추가 개발 근거

제공자 고정 커밋 cb64ded658b83ae97e6388dd373efad0e97bc47c와 원본 64 CSV의 Git blob을 읽기 전용으로 대조해 모두 일치함을 확인했다. 제공자 문서는 extor=회외, 60Hz 노치 및 20–450Hz 4차 Butterworth 처리를 명시한다. 구체적인 필터 계수·위상·실행 순서와 마커의 생리적 의미는 미확인이다. 결과는 `results/diagnostic_v2/provider_verification.json`, 추가 방법의 근거와 조건은 각 `experiments/*/PLAN.md`에 있다.
