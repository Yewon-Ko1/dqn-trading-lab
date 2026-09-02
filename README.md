# dqn-trading-lab

강화학습(DQN) 주식 매매 전략을 **바닥부터 직접 구현하며 검증하는** 개인 학습·실험 저장소입니다.
한이음 드림업 팀 프로젝트(AI 트레이딩 시스템)의 모델 파트를 맡으면서, 모델의 모든 부분을
스스로 설명할 수 있는 상태로 만들기 위해 데이터 → 피처 → 매매 환경 → DQN → 실험까지
가장 작은 형태로 다시 만들어 갑니다. 결과 논문 실험(Basic DQN vs Double DQN vs 기준선 4종,
walk-forward)도 이 저장소의 코드로 수행합니다.

## 원칙

1. **이해하지 못한 코드는 커밋하지 않는다.** 각 단계는 개념 정리(lesson.md) → 직접 구현(# TODO 채우기) → 자동 검증(check_XX.py) 순서로 진행하고, check가 전부 통과해야 다음 단계로 간다.
2. **모든 실험은 기준선과 비교한다.** Buy&Hold, MA교차, 현금 100%, 고정 노출 50%. 거래비용(수수료 0.015%, 매도세 0.25%) 포함.
3. **재현 가능해야 한다.** 시간 순 분할(셔플 금지), 시드 고정, 결과는 experiments.csv에 실행 인자와 함께 기록, 시드 5개 평균 ± 표준편차로 보고.

## 진행 상황

| 주차 | 단계 | 내용 | 상태 |
|---|---|---|---|
| 0주 | 00~01 | 일봉 데이터 수집, 피처 8종(수익률·이동평균 이격도·변동성) | 🔨 진행 중 |
| 1주 | 02~03 | 시간 분할·정규화(룩어헤드 방지), 기준선 4종 + 성과지표(Sharpe·Sortino·MDD) | ⏳ |
| 2주 | 04~06 | 매매 환경(state/action/reward), 랜덤 정책 검증, PyTorch Q네트워크 | ⏳ |
| 3주 | 07 | 표준 DQN(리플레이 버퍼·타겟 네트워크·ε-greedy·매 스텝 TD) | ⏳ |
| 4주 | 08~09 | Double DQN, walk-forward 실험(4 fold × 5 seed), 결과 정리 | ⏳ |

주차별 기록은 [`weekly/`](weekly/)에, 실험은 🧪 Experiment 이슈로 시작해 PR로 닫습니다.

## 핵심 결과 (실험 완료 후 갱신)

> 4개 fold walk-forward, 시드 5개 평균 ± 표준편차. (준비 중)

| 전략 | 누적수익률 | B&H 대비 초과 | Sortino | MDD | 거래 수 |
|---|---|---|---|---|---|
| Buy & Hold | — | — | — | — | — |
| Basic DQN | — | — | — | — | — |
| Double DQN | — | — | — | — | — |

## 실행

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install pandas numpy matplotlib finance-datareader   # torch는 06단계부터
python study/00_data/get_data.py                # 삼성전자 일봉 → out/005930.csv
python study/01_features/check_01.py            # 01단계 자동 채점
```

## 참고

- 팀 프로젝트: 2026 한이음 드림업 "강화학습 기술을 이용한 AI 트레이딩 시스템의 구현" (팀 저장소는 비공개)
- 학습 참고: quantylab/rltrader (책 『파이썬을 이용한 딥러닝/강화학습 주식투자』 코드) — 단, 본 저장소의 DQN은 rltrader의 에피소드 단위 학습이 아니라 Mnih et al.(2015)의 표준 DQN(리플레이 버퍼·타겟망)으로 구현
