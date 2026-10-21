# 출처와 적용 범위

[Pan et al., EMBC 2015](https://pubmed.ncbi.nlm.nih.gov/26737970/), DOI10.1109/EMBC.2015.7320070. 2026-10-21에 Europe PMC 공식 REST API의 PMID26737970 core 응답에서 초록 전체를 읽었다. PubMed 웹 응답은 제한됐고 IEEE 본문은 접근 확인 화면을 반환했다. Europe PMC는 전문을 구독 필요로 표시했다. 따라서 초록에 명시된 6차 AR와 잔차 비교, 대상 집단별 효과 차이까지만 근거로 삼았다. 전문의 추정법·정규화·분할·정확한 잔차 특징은 확인하지 못했다.

원연구는 AR 계수에 잔차를 추가했을 때의 정보를 조사했다. 현재 프로젝트에는 RMS 등 진폭 특징이 이미 있으므로 잔차가 추가 이득을 준다고 가정하지 않는다. 계수만 더한55D와 잔차까지 더한60D를 분리해 그 가설을 기각할 수 있게 했다. 최소제곱 계수와 잔차 RMS는 PLAN.md에 정한 이 프로젝트의 구현 선택이다.

이 연구는 강의와 연결된 DOI10.1007/s13534-026-00607-7과 다르다. 어느 논문에 대해서도 전문 재현 성공을 주장하지 않는다.

기존 코드의 중복 확인 범위: experiments/diagnostic_v2/features_v2.py, cross_spectrum_v6/features_cross.py, fine_spectrum_v8/features_fine.py와 experiments 전체의 AR·autoregressive·LDA·LinearSVC 검색. 기존40ms×5 subRMS는 시간 구간 요약을 이미 제공한다. diagnostic_v2/inner_summary.csv에서 time58.3644%, 기본58.4360%를 확인했다. 50ms×4 log 변화가 수학적으로 완전히 같은 것은 아니지만 추가 효과 근거가 약해 이번 고정 대조에서 제외했다.

내부 오류쌍 판단은 v10의 inner_fine64 및 inner_narrow64 각3개 검증 결과만 사용했다. 가장 큰 양방향 오류쌍은 fine에서(0,3)/(3,7)/(6,7), narrow에서(5,6)/(3,7)/(6,7)로 바뀌었다. 각 fold 전체 오류 중 최대쌍 비중은8.99–13.70%였다. 안정적으로 한 쌍에 집중된다는 조건을 충족하지 않아 쌍별 보정기를 만들지 않았다. 이는 모든 확률 보정이 무효라는 증거는 아니다.
