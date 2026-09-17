# DQN Trading Lab — 강화학습 투자전략 검증

[![CI](https://github.com/Yewon-Ko1/dqn-trading-lab/actions/workflows/ci.yml/badge.svg?branch=study)](https://github.com/Yewon-Ko1/dqn-trading-lab/actions/workflows/ci.yml)

강화학습 모델의 수익률을 단순히 높이는 것보다, **시장 데이터에서 투자전략을 어떻게 공정하게 검증하고 실패 원인을 추적할 것인가**를 실험하는 프로젝트입니다.

2026 한이음 드림업 팀 프로젝트 「강화학습 기술을 이용한 AI 트레이딩 시스템의 구현」에서 모델 파트를 담당하고 있습니다. 팀 시스템과 별도로, 데이터 수집부터 피처·매매 환경·DQN·walk-forward 평가까지 직접 재구현해 모델의 작동 방식과 한계를 검증하고 있습니다.

## 프로젝트에서 확인하려는 것

- Basic DQN과 Double DQN이 거래비용을 포함한 일봉 매매에서도 기준선을 넘어서는가?
- 성능 차이가 모델 구조, 입력 피처, 학습량보다 **평가 편향이나 과잉매매**에서 발생하지는 않는가?
- 하락장 방어와 상승장 참여 사이의 상충을 재현 가능한 실험으로 설명할 수 있는가?

## 현재 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 데이터 | 삼성전자·KODEX 200 일봉 수집, 액면분할 거래량 보정, 매매정지일 정제 | 완료 |
| 피처 | 수익률·추세·변동성·시장·수급·해외지수 피처와 룩어헤드 검사 | 완료 |
| 환경 | 매수·매도·보유, 거래비용, 포지션 상태, 최소 보유기간 | 완료 |
| 모델 | PyTorch Q-network, Replay Buffer, Target Network, ε-greedy, Basic/Double DQN | 완료 |
| 검증 | 기준선 4종, walk-forward, 다중 시드, Sharpe·Sortino·MDD | 완료 |
| 튜닝 | 검증 폴드 3개 × 시드 20개 자동 실험 | 진행 중 |
| 최종 평가 | 설정 동결 후 2021~2025 및 2026 보조 구간 1회 평가 | 예정 |

최종 평가 구간은 튜닝에 사용하지 않습니다. 검증 결과가 기대보다 나쁘더라도 평가 구간을 본 뒤 설정을 다시 고르지 않는 것이 원칙입니다.

## 핵심적으로 확인한 것

### 1. 좋은 결과보다 평가 구간의 정합성이 먼저였다

초기 실험에서 에이전트는 테스트 연도의 첫 관측창만큼 거래를 건너뛰었지만 Buy & Hold는 연초부터 평가되는 오류를 발견했습니다. 관측창 길이에 따라 유·불리가 달라져 오류 수정 전 window 비교와 2021~2024 결과를 최종 근거에서 제외했습니다.

현재는 직전 거래일을 워밍업으로 붙여 모든 전략을 같은 날짜 구간에서 평가하며, 코드에서 평가 시작일과 길이를 검사합니다. 수정 과정과 무효 처리 범위는 [`decisions.md`](decisions.md)에 기록했습니다.

### 2. 피처를 늘리는 것보다 과잉매매를 줄이는 효과가 컸다

기술지표, 장기 추세, 시장지수, 투자자 수급, 변동성 국면, 미국 지수 등 여러 피처 구성을 비교했지만 복잡한 입력이 안정적인 개선으로 이어지지는 않았습니다. 새 시드에서 후보 순위가 뒤집혀 20시드 기준 차이가 1.4%p로 줄었고, 단순한 5개 피처를 유지했습니다.

가장 큰 변화는 최소 보유기간 제약에서 나왔습니다. 검증 폴드 3개 × 시드 20개에서 평균 B&H 대비 초과수익률은 -20.8%에서 -9.2%로 개선됐고, 연평균 거래는 63회에서 29회로 줄었습니다. 여전히 B&H를 넘지는 못했지만, 부진의 주요 원인이 입력 부족보다 과잉매매에 가까웠다는 근거가 됐습니다.

현재는 이 작은 피처 세트와 거래 제약을 기준으로 남은 설정을 자동 체인에서 재검증하고 있습니다.

### 3. 결과를 보고 규칙을 바꾸지 않도록 결정 과정을 남겼다

- 튜닝 구간: 2018년 하락장, 2019년 상승장, 2020년 급락 후 반등장
- 평가 구간: 2021~2025년, 2026년은 부분 연도 보조 결과
- 후보 변경 조건: 평균 초과수익률, 폴드별 개선, 2018년 방어력 세 조건을 모두 충족
- 모든 설정과 결과를 [`out/tuning.csv`](out/tuning.csv)에 기록
- 가설 → 실험 → 결과 → 결정 근거를 [`decisions.md`](decisions.md)에 기록

자동 체인은 완료된 연도·시드 실행을 재사용하며, 단계별 판정은 [`out/auto_chain.md`](out/auto_chain.md)에 남깁니다.

현재 공개된 튜닝 수치는 **검증 구간의 중간 결과**입니다. 평가 구간의 최종 수치와 그림은 설정 동결 후 별도로 갱신합니다. 평가 정렬 오류 수정 전에 생성한 결과는 [`out/archive/pre_warmup_fix/`](out/archive/pre_warmup_fix/)로 분리했습니다.

## 평가 설계

- **시간 순 분할:** 셔플 없이 학습 구간 다음 연도를 테스트하는 walk-forward 방식
- **기준선:** Buy & Hold, MA20/60 교차, 현금 100%, 고정 노출 50%
- **거래비용:** 매수·매도 수수료 0.015%, 매도 거래세 0.25%
- **반복성:** 같은 설정을 여러 시드로 반복하고 평균과 시드 간 편차를 함께 보고
- **평가지표:** 누적수익률, B&H 대비 초과수익률, Sharpe, Sortino, MDD, 거래 횟수
- **비교 모델:** Basic DQN과 행동 선택·평가를 분리한 Double DQN

## 구현 흐름

```text
OHLCV 수집·정제
  → 룩어헤드 없는 피처 생성
  → 시간 순 walk-forward 분할
  → 거래비용을 반영한 TradingEnv
  → Basic DQN / Double DQN 학습
  → 기준선과 동일 구간 평가
  → 설정·시드·성과를 CSV로 기록
```

학습용 코드는 개념 정리(`lesson.md`) → 직접 구현 → 단계별 검사(`check_XX.py`) 순서로 구성했으며, 핵심 검사는 CI에서도 실행합니다.

## 주요 파일

| 경로 | 역할 |
|---|---|
| [`study/01_features/`](study/01_features/) | 기본·확장 피처 생성과 룩어헤드 검사 |
| [`study/04_env/trading_env.py`](study/04_env/trading_env.py) | 매매 환경과 거래비용·보유 제약 |
| [`study/07_dqn/dqn.py`](study/07_dqn/dqn.py) | Replay Buffer, DQN, Double DQN |
| [`study/08_runner/run_experiment.py`](study/08_runner/run_experiment.py) | walk-forward 본실험 실행기 |
| [`study/09_tuning/tune.py`](study/09_tuning/tune.py) | 설정별 검증 실행과 재개 기능 |
| [`study/09_tuning/auto_chain.py`](study/09_tuning/auto_chain.py) | 사전 정의된 규칙에 따른 자동 튜닝 체인 |
| [`decisions.md`](decisions.md) | 설계 선택, 실패 실험, 오류 수정 기록 |
| [`weekly/`](weekly/) | 데이터 구축부터 검증 자동화까지의 주차별 진행 기록 |

## 실행 예시

Python 3.11 이상을 기준으로 합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 단계별 검사
python study/01_features/check_01.py
python study/04_env/check_04.py
python study/07_dqn/check_07.py

# 튜닝 재개: 완료된 연도·시드는 건너뜀
python study/09_tuning/auto_chain.py --skip-final
```

## 범위와 한계

- 이번 논문 실험은 삼성전자 단일 종목 사례 연구이며, KODEX 200은 시장지수 보조 결과로만 사용합니다.
- 행동은 매수·매도·보유이고 포지션은 long/cash로 제한됩니다.
- 백테스트 결과이며 실제 운용 성과를 의미하지 않습니다.
- 다종목 일반화, 더 다양한 보상 설계, 국면별 모델 전환은 후속 과제로 남겨 두었습니다.

## 참고

- 팀 프로젝트: 2026 한이음 드림업 「강화학습 기술을 이용한 AI 트레이딩 시스템의 구현」(팀 저장소 비공개)
- 학습 참고: quantylab/rltrader, Mnih et al. (2015), van Hasselt et al. (2016)
- 본 저장소의 DQN은 에피소드 종료 후 일괄 학습하는 구조가 아니라 Replay Buffer와 Target Network를 사용하는 매 스텝 TD 학습으로 구현했습니다.
