# 07. DQN — rltrader의 DQN에서 표준 DQN으로

> **rltrader에서는:** `learners.py`의 `DQNLearner`가 있었다. 이름은 DQN이지만 Mnih et al.(2015)의 DQN과는 구조가 다르다.
> 이 단계의 목표는 "rltrader DQN이 무엇을 생략했는지"를 정확히 알고, 생략된 네 가지를 채워 넣어 표준 DQN을 만드는 것이다.
> 논문에서 "DQN"이라고 쓰면 심사자는 표준 DQN을 기대한다. 팀 코드 `backend/app/rl/agents/dqn.py`도 표준 쪽이다.

## 0. 먼저 rltrader의 DQNLearner를 다시 읽어 보면

`learners.py` 371~394행. 흐름은 이렇다.

1. `run()`이 한 에피소드(학습 구간 전체)를 처음부터 끝까지 돈다. 매일 `value_network.predict`로 Q값 3개를 뽑고 ε-greedy로 행동한다.
2. 매일의 (sample, action, value, reward=손익률)을 `memory_*` 리스트에 **쌓기만** 한다. 학습은 안 한다.
3. 에피소드가 끝나면 `get_batch()`가 메모리를 **거꾸로** 훑으면서 목표값을 만든다:

```python
for i, (sample, action, value, reward) in enumerate(reversed(memory)):
    r = self.memory_reward[-1] - reward          # "마지막 손익률 − 그 날 손익률"
    y_value[i] = value                           # 나머지 행동의 목표 = 예측값 그대로
    y_value[i, action] = r + discount * value_max_next
    value_max_next = value.max()                 # 다음 날 Q의 max = 그 날 '예측했던' 값
```

4. `fit()`이 이 배치 전체로 `train_on_batch` 한 번. 다음 에피소드로.

이걸 표와 비교하면 차이가 보인다.

## 1. 차이 네 가지

| | rltrader `DQNLearner` | 표준 DQN (Mnih 2015) | 왜 중요한가 |
|---|---|---|---|
| **① 언제 학습하나** | 에피소드가 끝난 뒤 한 번 | **매 스텝**(또는 몇 스텝마다) 미니배치로 | 1년치 에피소드 한 번에 한 번 갱신하면 학습이 너무 느리고, 배치 안 샘플이 전부 같은 정책에서 나와 편향된다 |
| **② 무엇으로 학습하나** | 방금 끝난 에피소드의 메모리 (순서대로) | **리플레이 버퍼**에서 무작위로 뽑은 과거 transition | 연속된 날짜 샘플은 서로 강하게 상관되어 있다. 무작위 추출이 이 상관을 깨서 학습을 안정시킨다 |
| **③ 목표값(TD target)** | `r + γ · (이전 루프에서 본 예측값의 max)` — 같은 네트워크가 방금 뱉은 값 | `r + γ · max_a Q_target(s', a)` — **따로 얼려 둔 타겟 네트워크**가 계산 | 학습 중인 네트워크로 목표를 만들면 목표가 학습할 때마다 움직여 발산하기 쉽다. 타겟망을 N스텝마다만 복사해 목표를 고정한다 |
| **④ 보상** | `마지막 손익률 − 그 날 손익률` (에피소드가 끝나야 알 수 있는 지연 보상) | **그 날의 자산 변화율** (즉시 보상) | 지연 보상은 "결국 어떻게 됐나"만 알려줘서 어느 행동이 기여했는지 구분이 안 된다. 즉시 보상 + 할인 누적이 표준 |

정리하면 rltrader는 "에피소드 단위 몬테카를로 회귀"에 가깝고, 표준 DQN은 "스텝 단위 TD 학습"이다. 둘 다 Q값을 신경망으로 근사한다는 점만 같다.

## 2. 표준 DQN 한 스텝의 흐름

```
state s ──► Q_online(s) ──► ε-greedy ──► action a
                                          │
                     env.step(a) ──► reward r, next_state s', done
                                          │
                     buffer.push(s, a, r, s', done)          ← ② 리플레이 버퍼
                                          │
        (buffer가 충분히 차면) batch = buffer.sample(64)      ← ① 매 스텝 학습
                                          │
        target = r + γ · max_a' Q_target(s', a') · (1 − done)  ← ③ 타겟망
        loss   = Huber( Q_online(s)[a], target )
        loss.backward(); optimizer.step()
                                          │
        (N 스텝마다) Q_target ← Q_online 복사                   ← ③ 타겟망 갱신
        ε ← max(ε_min, ε · decay)
```

## 3. 과제 — 네 조각을 순서대로 채운다

`dqn.py`의 TODO는 위 네 가지에 하나씩 대응한다. **한 조각을 채울 때마다 check_07.py를 돌린다.** 조각별로 검사가 따로 있다.

| TODO | 만드는 것 | check가 확인하는 것 |
|---|---|---|
| 1 | `ReplayBuffer.push / sample` | 용량 넘으면 오래된 것부터 버림, sample이 무작위(같은 인덱스 반복 없음) |
| 2 | `select_action` (ε-greedy) | ε=1이면 행동 분포가 균등, ε=0이면 argmax |
| 3 | `compute_target` (타겟망 사용, done 처리) | done=True인 샘플의 target이 r 과 같음, 타겟망 파라미터가 학습 중 변하지 않음 |
| 4 | `train_step` (Huber 손실, backward, N스텝마다 타겟 복사) | 같은 배치로 두 번 학습하면 손실이 줄어듦, 시드 고정 시 완전 재현 |

그 다음 `train.py`로 삼성전자 1개 fold를 15 에피소드 학습하고, 에피소드별 누적 보상 곡선이 오르는지 본다 (AT-06).

## 4. 07이 끝나면 할 것

- 팀 코드 `backend/app/rl/agents/dqn.py`를 열어 `train_step`을 내 것과 나란히 놓고 읽는다. 같은 네 조각이 있을 것이다. 다른 점을 notes.md에 적는다.
- `DoubleDQNAgent`(같은 파일 197행~)를 본다. 표준 DQN에서 **③의 한 줄만** 바뀐다:
  `max_a' Q_target(s', a')` → `Q_target(s', argmax_a' Q_online(s', a'))`. 이게 08~09단계의 M2 모델이다.
- 논문 3장 "제안 모델"의 DQN 설명 문단은 이 lesson의 2절 그림을 글로 옮기면 된다.

## 자주 하는 실수

- 타겟 계산에 `torch.no_grad()`를 안 씌워서 타겟망까지 그래디언트가 흐름.
- `done=True`인데 `γ · max Q(s')`를 더함 → 마지막 날 목표값이 이상해짐.
- 버퍼가 배치 크기보다 작을 때 sample을 호출해서 에러.
- 테스트(평가) 때 ε을 0으로 안 놓고, 버퍼에 push하거나 train_step을 호출함 (MDL-08 위반).
