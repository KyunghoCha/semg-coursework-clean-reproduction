# 근거와 접근 범위

강의의 실험 조건은 [슬라이드 검토 기록](LECTURE_REVIEW.md)과 [재현 명세](PROTOCOL.md)에서 추적한다. 아래 외부 자료는 API와 검증 방법을 확인하는 데 사용했다. 기존 프로젝트의 코드·결과·문서는 구현 근거로 읽거나 가져오지 않았다.

| 자료 | 읽은 범위와 적용 |
|---|---|
| [논문 공개 페이지](https://link.springer.com/article/10.1007/s13534-026-00607-7) | 제목·초록·Data availability·구독 안내. 8명/5채널/8클래스와 RF96.72±.58%는 초록에서 확인. 방법 전문은 구독 콘텐츠여서 미확보 |
| [scikit-learn 데이터 누수](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | fit/transform, 학습 부분에서만 전처리 통계 계산. 각 split의 scaler fit 검증에 반영 |
| [scikit-learn 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html) | 테스트로 모델 선택할 때의 위험, 그룹 단위 분할. 시행 내부 창을 같은 집합에 유지 |
| [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html) | device=cuda와 hist 사용법. 최신 문서의 CUDA 최소 버전을 설치된3.0.5에 그대로 대입하지 않고 로컬 build_info와 실제 GPU 학습으로 확인 |
| [SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html) | 단측 PSD, 샘플링, 창, detrend, axis 정의. MNF/MDF 확장 설정을 명시적으로 고정 |

검색어는 `cross validation grouped data`, `overlapping samples`, `StandardScaler data leakage`, `XGBoost GPU Pascal compute capability`를 사용했다. 초기 검색의 무관한 capability 결과는 근거에서 제외했다. XGBoost release_3.0 문서 주소는 도구에서 열리지 않아 설치된 패키지의 설정과 합성 GPU 실행을 버전별 증거로 남겼다.

한국어 문서는 `korean-technical-writing/SKILL.md`와 참고7개(문서 워크플로, 용어 정책, 문장 진단, 연구 문체, 교수 논문 문체, 관점별 검토, 프로젝트 예시)를 읽고 작성했다. 독자는 과제 결과와 조건을 확인하려는 수업 담당자와 재실행자다. 본문은 문제·조건·관찰·해석을 구분하고 수치표와 실제 집계그림을 연결한다.

## 2차 진단에서 추가 확인한 제공자 문서

[데이터 제공자 README, 고정 커밋 cb64ded](https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset/blob/cb64ded658b83ae97e6388dd373efad0e97bc47c/README.md): 2026-10-20 UTC에 읽었다. 공개 Git tree의 blob 해시와 로컬 원본 64개의 바이트를 Git blob 형식으로 해시한 결과가 모두 일치했다. `extor=회외`, `pro=회내`, 5개 채널 순서, 60Hz 노치·20–450Hz 4차 Butterworth라는 제공자 명세를 확인했다. 실제 필터 계수·실행 순서·위상과 마커의 생리적 정렬은 여전히 확인되지 않았다. 초기 실험의 명세와 결과를 소급 변경하지 않고 현재 README의 확인 범위를 갱신했다.


## v20–v22의 확률 보정과 계산 경로

설치된 scikit-learn1.7.2의 [SVC](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html), [SVM의 확률 설명](https://scikit-learn.org/1.7/modules/svm.html), [CalibratedClassifierCV](https://scikit-learn.org/1.7/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)와 실제 설치 소스를 확인했다. 내장 SVC 확률 보정의 기본 분할을 그대로 사용하지 않고 학습 시행을 층화해 만든 인덱스를 별도 보정 CV로 전달했다. ensemble=False의 보정 OOF 및 전체 학습 재학습을4개의 실제base fit로 확인했다.

후속 후보 비교에서는 [PCA1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html)의 중심화·성분 선택·whitening 설명과 [ExtraTrees1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)의 min_samples_leaf·복잡도 제어·CPU 병렬 설명을 읽었다. API상 가능한 방법과 현재 자료에서 성능이 확인된 방법을 구분했다. 이 두 후보는 v21의 대안 비교이며 실행했다고 기록하지 않는다.

v22 호출 생략의 근거는 새 논문 성능 수치가 아니라, 확률 가중합의 차이 하한 `(1-beta)*delta-beta`다. 이 값이 양수이면 모든 다른 클래스보다 기준 최대 클래스가 크다. 실제 구현은 수치 여유와 동률 처리를 보수적으로 두고, 분류 라벨만 보존하는 경로로 제한한다. 전체 결합 확률·신뢰도를 보존한다는 주장은 하지 않는다.

## 2026-10-22 논문 재대조

공식 공개 초록·Data availability와 공식 Fig.5 픽셀을 다시 확인했다. 각 행48개, 총384개이며 전사한 집계와 산술은 `results/paper_comparison/official_fig5_counts.json`에 남겼다. 논문 전문을 확보했다는 뜻은 아니다. 강의의 모델 코드·구간 가정과 대조한 결론은 `PAPER_IMPLEMENTATION_COMPARISON.md`에 있다. 제공자 저장소와 원본64개가 일치한다는 기존 검증과 논문 원자료 동일성은 별개의 주장으로 구분했다.
