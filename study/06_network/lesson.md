# 06. PyTorch 최소 신경망 — state를 넣으면 Q값 3개가 나오는 상자

> **rltrader에서는:** `networks_pytorch.py`의 DNN 클래스. Linear 층을 쌓고 forward로 통과시키는
> 구조가 우리가 만들 것과 같다. 팀 backend의 `rl/networks/dnn.py`도 동일.
> 시작 전에:  pip install torch   (용량이 커서 몇 분 걸림 — venv 켠 상태에서!)

## 1. 텐서(tensor) = 미분이 되는 numpy 배열

PyTorch의 기본 재료. 이미 아는 numpy 배열과 거의 같은데, 두 가지가 추가된다:
"이 값이 어떤 계산을 거쳐 왔는지 기억"(자동 미분)과 GPU 계산. 변환도 자유롭다.

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])      # 만들기
x.numpy()                               # numpy 로
torch.tensor(np_array)                  # numpy 에서
```

우리 파이프라인: 환경의 state(numpy) → `torch.tensor(state)` → 신경망 → Q값(텐서) → `.argmax()` 행동.

## 2. 신경망 = "행렬곱 + 꺾기"를 쌓은 것

**`nn.Linear(163, 64)`** — 입력 163개를 출력 64개로 바꾸는 층. 내부는 `출력 = W×입력 + b`
(W는 163×64 행렬, b는 64개). 이 W와 b가 **학습되는 숫자들(파라미터)**이고, 처음엔 난수다.

**`nn.ReLU()`** — 음수를 0으로 꺾는 함수. 이 "꺾임"이 없으면 Linear를 백 층 쌓아도
결국 직선 하나라, 곡선(복잡한 패턴)을 못 배운다. 층 사이에 끼워 비선형성을 준다.

**`nn.Sequential(...)`** — 층들을 순서대로 통과시키는 파이프. 우리 Q네트워크:

```
state 163개 → Linear(163→64) → ReLU → Linear(64→64) → ReLU → Linear(64→3) → Q값 3개
                                                                  (BUY, SELL, HOLD)
```

마지막 층 뒤에 ReLU가 **없는** 이유: Q값은 "예상 누적수익"이라 음수여야 할 때가 있다.
ReLU를 씌우면 손실 상황을 표현 못 한다.

## 3. 학습 = 4박자 반복

신경망 학습은 어디서나 이 네 줄이다. DQN(07)도 이 뼈대에 목표값 계산만 얹는다.

```python
pred = model(x)                    # ① forward: 예측
loss = F.smooth_l1_loss(pred, y)   # ② loss: 정답과의 거리 (숫자 하나)
optimizer.zero_grad()              #    (직전 기울기 청소 — 안 하면 누적됨!)
loss.backward()                    # ③ backward: 각 파라미터가 loss 에 얼마나 책임 있는지(기울기) 계산
optimizer.step()                   # ④ step: 기울기 반대 방향으로 파라미터를 조금 이동
```

비유: 산에서 안개 속 하산. loss = 현재 고도, backward = 발밑 경사 재기, step = 경사 아래로 한 걸음.
optimizer(Adam)는 "보폭을 얼마나, 어느 방향으로"를 관리하는 등산 가이드이고, lr(학습률)은 기본 보폭.

`smooth_l1_loss`(Huber)는 예측-정답 차이가 작으면 제곱, 크면 절댓값으로 재는 손실 함수 —
가끔 튀는 값(급등락일)에 학습이 휘둘리지 않아 DQN 표준이다.

## 4. 이번 단계에서 확인할 것

아직 매매와 무관한 **장난감 문제**로 4박자를 익힌다: 랜덤 state를 주고, 정답 Q값을
우리가 정한 간단한 규칙으로 만들어서, 네트워크가 그걸 흉내 내게 학습시킨다.
"loss가 실제로 내려가는가"를 눈으로 보는 게 목표다. 07에서는 정답 Q값 자리에
`r + γ·maxQ_target`(TD 목표)이 들어갈 뿐, 나머지는 오늘 것 그대로다.

## 과제

`qnet.py`의 TODO 3개. `python study/06_network/check_06.py` 전부 OK면 통과.

## 자주 하는 실수

- 마지막 층 뒤에 ReLU를 붙임 (Q값이 음수가 못 됨)
- `zero_grad()` 누락 → 기울기가 계속 누적돼 학습이 산으로 감
- loss.backward() 와 optimizer.step() 순서 뒤바꿈
- 입력을 float64(numpy 기본)로 넣어 dtype 에러 — `dtype=torch.float32` 또는 `.float()`
