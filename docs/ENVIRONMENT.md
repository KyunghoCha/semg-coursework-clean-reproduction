# 실행 환경

사용자 PC에서 CPU 모델과 GPU 모델을 함께 실행했다. XGBoost는 TITAN Xp(12GB)에서 CUDA histogram을 사용했다. 시스템 환경과 다른 프로세스는 변경하지 않았다.

| 항목 | 실행 값 |
|---|---|
| Python | 3.13.9, Anaconda 배포 |
| NumPy / pandas | 2.3.5 / 2.3.3 |
| SciPy / scikit-learn | 1.16.3 / 1.7.2 |
| XGBoost | 3.0.5, CUDA build12.8 |
| matplotlib / pytest | 3.10.6 / 8.4.2 |
| joblib | 1.5.2 |
| CPU | Intel Core i7-8700 @3.20GHz, 6코어/12스레드 |
| GPU / 드라이버 | NVIDIA TITAN Xp 12288MiB / 580.178.04 |
| 학습 seed | 분할42, RF42, XGB42; SVC probability=False는 결정적 fit |
| GPU 확인 | 합성80×25 입력, 별도2-round smoke test 후 device=cuda:0 확인 |

허용된 기존 XGBoost 가상환경의 설치 패키지와 Python 실행 파일만 재사용했다. 해당 가상환경은 `include-system-site-packages=true`이며, editable 설치 프로젝트는 없다. 기존 프로젝트의 Python 모듈은 가져오지 않았고 `PYTHONPATH`는 새 프로젝트 `src`로 지정했다. 최초 기준·개선 학습에서는 패키지를 새로 설치하지 않았다. 이후 GPU 호출 경로 대조를 위해 아래의 별도 환경을 추가했다. 새 환경에서 실행하려면 requirements.txt로 별도 venv를 만든다.

`private/environment_baseline.json`과 `private/environment_improve.json`은 실제 실행 환경 및 XGBoost 빌드 정보를 담는다. 각 fit의 booster 설정, scaler 평균·분산·표본 수는 `private/<phase>/*_parameters.json`에 기록한다. GPU 접근은 샌드박스 밖 호스트 실행을 통해 얻었다. 일반 터미널에서는 이 추가 권한 단계가 필요하지 않다.

최초 결과표의 추론 시간은 이미 표준화된 창1개를 100회 호출한 평균이다. XGB는 CPU 배열을 DMatrix로 만드는 시간과 GPU 전송을 포함하며, RF는 n_jobs=-1의 단일창 호출 오버헤드를 포함한다. 하드웨어·라이브러리·배치 크기가 다른 논문 수치와 직접 속도 순위를 비교할 수 없다. 신호 취득2초, 특징 추출, scaler transform은 이 추론 시간에 포함하지 않았다.


추가 비용 비교는 `experiments/cost_benchmark/PLAN.md`의 별도 절차다. threadpoolctl 3.5.0으로 라이브러리 스레드 상한을 6으로 고정하고 모델마다 새 프로세스를 사용한다. 단일 창 200회와 batch256 30회의 중앙값·p95·처리량을 실제 원신호의 특징 추출부터 측정한다. GPU 호출 전후에는 CUDA 동기화를 확인한다. 최초 표의 평균 추론 시간과 이 새 측정값을 같은 조건으로 비교하지 않는다. 신호 길이·과거 문맥과 계산 지연도 구분한다.


추가 추론 환경은 `private/inference_env`에 새로 만들었다. 공식 PyPI의 CuPy CUDA12용 13.6.0, fastrlock 0.8.4, nvidia-nccl-cu12 2.32.3과 고정한 XGBoost 3.0.5를 해당 환경에 설치했다. 기존 환경·시스템 패키지는 변경하지 않았다. 핵심 NumPy/pandas/SciPy/scikit-learn/joblib/XGBoost 버전 및 `libxgboost.so`의 SHA-256이 학습 환경과 같음을 확인했다. XGBoost 빌드의 CUDA 12.8과 실행 시 CuPy가 보고한 CUDA runtime 12.9를 구분한다. 실제 GPU compute capability는 6.1이었다. [환경 대조](../results/cost_benchmark/inference_environment.json)에 확인값을 남겼다.

호출 경로 추가 대조는 기존 비용 결과를 보존하고 단일 창 100회·batch256 10회로 실행했다. 동일 수식의 특징 벡터화는 CPU에서 원래/새 경로를 번갈아 각 200회 측정했다. 원래 32조합과 이 후속 비교는 표를 분리했다. 각 방법의 정확한 반복 수와 warmup은 `experiments/cost_benchmark/`의 계획 및 결과 JSON에 있다.
