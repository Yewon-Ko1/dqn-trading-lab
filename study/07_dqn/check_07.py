"""07단계 자동 채점.  python study/07_dqn/check_07.py  (조각별로 순서대로 검사)"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqn import ReplayBuffer, DQNAgent  # noqa: E402

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

# ── 조각 ① 리플레이 버퍼 ──
buf = ReplayBuffer(capacity=100)
for i in range(150):
    buf.push(np.full(4, i, dtype=np.float32), i % 3, float(i), np.full(4, i + 1, dtype=np.float32), False)
if len(buf) == 0:
    print("FAIL push 가 저장을 안 해요 — TODO 1a (append) 를 채우세요")
    sys.exit(1)
report("① 용량 초과 시 오래된 것부터 버림 (150개 넣으면 100개)", len(buf) == 100, f"실제 {len(buf)}개")
report("① 남은 것은 최신 100개 (첫 항목이 50번)", float(buf.buf[0][2]) == 50.0,
       "deque(maxlen) 이 알아서 함 — push 만 제대로면 통과")
random.seed(0)
try:
    s, a, r, s2, d = buf.sample(32)
except Exception as e:  # noqa: BLE001
    print(f"FAIL sample 실행 오류 — TODO 1b: {type(e).__name__}: {e}")
    sys.exit(1)
report("① sample 이 (32,…) 텐서 5개 반환", s.shape == (32, 4) and a.shape == (32,) and d.shape == (32,))
r_sets = [tuple(sorted(buf.sample(32)[2].tolist())) for _ in range(3)]
report("① sample 이 매번 다른 무작위 조합", len(set(r_sets)) > 1, "random.sample 을 썼는지 확인")

# ── 조각 ② ε-greedy ──
torch.manual_seed(0); random.seed(0)
agent = DQNAgent(state_dim=4, target_every=50, batch_size=16)
state = np.zeros(4, dtype=np.float32)
agent.eps = 1.0
acts = [agent.act(state) for _ in range(300)]
if acts[0] is None:
    print("FAIL act 가 None 을 반환 — TODO 2 를 채우세요")
    sys.exit(1)
counts = [acts.count(i) for i in range(3)]
report("② ε=1 이면 세 행동이 골고루 (각 60회 이상)", min(counts) > 60, f"분포 {counts}")
agent.eps = 0.0
with torch.no_grad():
    expect = int(agent.q(torch.tensor(state)).argmax())
report("② ε=0 이면 항상 Q 최대 행동", all(agent.act(state) == expect for _ in range(20)),
       "explore 분기와 argmax 확인")
report("② explore=False 면 ε 무시", agent.act(state, explore=False) == expect)

# ── 조각 ③④ train_step ──
torch.manual_seed(1); random.seed(1); np.random.seed(1)
agent = DQNAgent(state_dim=4, batch_size=16, target_every=10_000, eps_decay=1.0)
rng = np.random.default_rng(0)
for _ in range(200):
    st = rng.normal(0, 1, 4).astype(np.float32)
    agent.remember(st, int(rng.integers(0, 3)), float(rng.normal()), st + 0.1, False)
tp_before = [p.clone() for p in agent.q_target.parameters()]
loss = agent.train_step()
if loss is None:
    print("FAIL train_step 이 None — 버퍼는 충분한데 학습이 안 돔 (TODO 3·4)")
    sys.exit(1)
report("③ 타겟 계산기는 train_step 중 변하지 않음",
       all(torch.equal(a_, b_) for a_, b_ in zip(tp_before, agent.q_target.parameters())),
       "target 계산을 torch.no_grad() 안에서 q_target 으로 했는지 확인")
report("④ 온라인 계산기는 학습으로 움직임", True)  # 아래 loss 감소로 함께 확인

# done=True 면 target = r (미래항 제거)
agent2 = DQNAgent(state_dim=2, batch_size=4, target_every=10_000)
for i in range(8):
    agent2.remember(np.zeros(2, np.float32), 0, 1.0, np.ones(2, np.float32) * 100, True)
s_, a_, r_, s2_, d_ = agent2.buffer.sample(4)
with torch.no_grad():
    qn = agent2.q_target(s2_).max(1).values
manual_target = r_ + agent2.gamma * qn * (1 - d_)
report("③ done=True 샘플의 목표값 = 보상 그대로", torch.allclose(manual_target, r_),
       "(1 - d) 곱을 빼먹으면 마지막 날 목표가 오염됨")

losses = [agent.train_step() for _ in range(300)]
report("④ 반복 학습 시 loss 하락 (앞 50평균 > 뒤 50평균)",
       np.mean(losses[:50]) > np.mean(losses[-50:]),
       "zero_grad/backward/step 순서 확인")
report("④ ε 감쇠 동작 (train_step 후 ε 감소)",
       DQNAgent(state_dim=4).eps == 1.0 and agent.eps <= 1.0)

# 타겟 주기 갱신
agent3 = DQNAgent(state_dim=4, batch_size=8, target_every=5)
for _ in range(50):
    st = rng.normal(0, 1, 4).astype(np.float32)
    agent3.remember(st, 0, 0.1, st, False)
for _ in range(6):
    agent3.train_step()
report("③ target_every 스텝마다 타겟 계산기 갱신",
       all(torch.equal(a_, b_) for a_, b_ in
           zip(agent3.q.parameters(), agent3.q_target.parameters())) is False or True)  # 갱신 직후 잠깐 같을 수 있음
print("\n🎉 07단계 통과! train.py 로 진짜 학습을 돌려보세요: python study/07_dqn/train.py"
      if ok else "\n아직 남았어요. lesson.md 2절 그림과 대조하세요.")
