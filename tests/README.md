# tests

검사는 학습 단계와 가까운 `study/*/check_XX.py`에 배치했습니다. 합성 데이터로 핵심 불변 조건을 확인하며 GitHub Actions에서도 같은 검사를 실행합니다.

주요 검사 항목은 다음과 같습니다.

- 피처의 룩어헤드와 결측치 처리
- 시간 순 분할과 학습 구간 기준 정규화
- 기준선과 성과지표 계산
- 거래비용, 매수·매도, 보상과 자산가치 갱신
- Q-network 출력과 학습
- Replay Buffer, ε-greedy, Target Network, Double DQN 목표값

외부 시장 데이터가 필요한 `study/01_features/check_extra.py`는 CI에서 제외하고 데이터 생성 후 로컬에서 실행합니다.
