# 08. 실험 실행기 — 지금까지 만든 전부를 논문 표 하나로

> 여기서부터는 "공부"가 아니라 **연구**다. 01(피처)·02(fold)·03(기준선·지표)·04(환경)·07(DQN)이
> 전부 부품으로 들어간다. 새 개념은 Double DQN 한 줄뿐이고, 나머지는 반복 실행과 기록의 규율이다.

## 1. 실험 규율 세 가지 (명세서 MDL-07·09·10)

1. **모든 실행은 experiments.csv 에 한 줄씩 쌓인다** — 언제, 어떤 모델, 어느 fold, 어떤 시드,
   에피소드 몇으로 돌렸고 지표가 얼마였는지. 논문 표의 모든 숫자는 이 파일의 행으로
   역추적 가능해야 한다. "이 숫자 어디서 났어요?"에 파일 한 줄로 답하는 것.
2. **같은 시드는 같은 결과** — run_experiment.py 는 시드를 고정하고 돌린다. 의심되면 같은
   명령을 두 번 돌려 experiments.csv 의 두 행이 일치하는지 보면 된다.
3. **보고는 항상 평균 ± 표준편차** — 시드 5개를 돌린 뒤 한 시드만 골라 말하는 순간 체리피킹이다.

## 2. Double DQN — 논문의 M2, 코드로는 두 줄

기본 DQN 의 TD 목표는 `max Q_target(s')` 인데, "고르는 것"과 "평가하는 것"을 같은(타겟)
계산기가 하다 보니 **우연히 높게 추정된 행동이 계속 뽑히는 과대평가**가 생긴다.
Double DQN(van Hasselt 2016)은 역할을 나눈다:

- 행동 **선택**: 온라인 계산기 `argmax Q(s')`
- 그 행동의 **평가**: 타겟 계산기 `Q_target(s')[선택된 행동]`

"내가 고르고, 심판이 점수 매긴다." 이 분리로 과대평가가 줄고, 트레이딩에서는
"다 좋아 보여서 마구 사고파는" 행동이 줄어드는 효과로 나타나는지가 우리 실험 질문이다.

### 과제 (dqn.py 에 직접 추가 — 이번 단계의 유일한 코딩)

1. `DQNAgent.__init__` 의 파라미터에 `double: bool = False` 를 추가하고 `self.double = double` 저장.
2. `train_step` 의 TD 목표 계산(TODO 3으로 채웠던 곳)을 분기로 바꾼다:

```python
with torch.no_grad():
    if self.double:                                   # M2: 선택은 온라인, 평가는 타겟
        next_a = self.q(s2).argmax(1, keepdim=True)
        q_next = self.q_target(s2).gather(1, next_a).squeeze(1)
    else:                                             # M1: 기존 그대로
        q_next = self.q_target(s2).max(1).values
    target = r + self.gamma * q_next * (1 - d)
```

`gather(1, next_a)` = "각 행에서 next_a 가 가리키는 열의 값을 꺼내라" — q_sa 계산에서 이미 본 함수다.
바꾼 뒤 `python study/07_dqn/check_07.py` 가 **여전히 전부 통과**해야 한다 (기본값 False 라 기존 동작 불변).

## 3. 실행기 사용법

```bash
# 빠른 점검 (fold 1개 × 시드 1개 × 에피소드 2 — 배관 확인용, 몇 분)
python study/08_runner/run_experiment.py --quick

# 본실험 (fold 4개 × 시드 5개 × 모델 2개 = 40회 학습 — 몇 시간, 밤에 걸어두기)
python study/08_runner/run_experiment.py

# 부분 실행도 가능
python study/08_runner/run_experiment.py --models dqn --seeds 1 2 3
```

끝나면: `out/experiments.csv`(전체 기록), `out/summary.md`(논문 표 초안: 평균±표준편차),
`out/fig_*.png`(fold 별 자산 곡선 — 시드 5개가 얼마나 흩어지는지 눈으로).

## 4. 결과를 읽는 법 (미리 마음의 준비)

- 절대수익률이 음수여도 실패가 아니다. 기준선 4개 대비 **어디서 이기고 어디서 지는지**가 결과다.
- 시드 표준편차가 크면 그것 자체가 발견이다 — "DQN 학습 불안정성"은 우리 논문의 후보 논점.
- fold 별로 갈리면(2022·2024 선방, 2023 열세) "방어형" 서사의 근거가 된다.
- 어떤 패턴이 나오든 experiments.csv 가 있으면 논문이 된다. 없는 것만이 실패다.
