"""07단계: 표준 DQN. lesson.md 를 먼저 읽는다 (rltrader DQN 과의 차이 4가지).

06의 build_qnet(계산기)과 4박자를 그대로 쓰고, 그 위에 네 조각을 얹는다:
  ① 리플레이 버퍼  ② ε-greedy  ③ 타겟 네트워크  ④ TD 목표로 매 스텝 학습

채점:  python study/07_dqn/check_07.py   (조각 하나 채울 때마다 돌려볼 것)
"""
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_network"))
from qnet import build_qnet  # noqa: E402


class ReplayBuffer:
    """과거 경험 (s, a, r, s', done) 을 저장했다가 무작위로 꺼내주는 통."""

    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)   # 꽉 차면 오래된 것부터 자동 삭제

    def push(self, s, a, r, s2, done):
        # TODO 1a. 튜플 (s, a, r, s2, done) 을 self.buf 에 추가하라 (append)
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        """무작위 batch_size 개를 뽑아 텐서 5개로 돌려준다."""
        # TODO 1b. random.sample(self.buf, batch_size) 로 무작위 추출
        batch = random.sample(self.buf,batch_size )
        s, a, r, s2, d = zip(*batch)        # 튜플들을 항목별로 분리
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int = 3, gamma: float = 0.99,
                 lr: float = 1e-3, batch_size: int = 64, target_every: int = 500,
                 eps_start: float = 1.0, eps_min: float = 0.05, eps_decay: float = 0.999,
                 double: bool = False, hidden: int = 64):
        self.q = build_qnet(state_dim, hidden=hidden, n_actions=n_actions)          # 온라인 계산기
        self.q_target = build_qnet(state_dim, hidden=hidden, n_actions=n_actions)   # 타겟 계산기(얼려둠)
        self.q_target.load_state_dict(self.q.state_dict())           # 처음엔 똑같이 맞춤
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.n_actions = n_actions
        self.gamma = gamma                  # 할인율: 내일 보상의 오늘 가치 (0.99)
        self.batch_size = batch_size
        self.target_every = target_every    # 몇 스텝마다 타겟 계산기를 갱신할지
        self.step_count = 0
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.double = double

    # ── ② ε-greedy ─────────────────────────────────────────────
    def act(self, state, explore: bool = True) -> int:
        """확률 ε 로는 아무 행동(탐험), 아니면 Q값이 최대인 행동(활용).

        explore=False 면(테스트) 항상 Q 최대 행동.
        """
        # TODO 2.
        #   if explore and random.random() < self.eps:  → random.randrange(self.n_actions)
        #   아니면: torch.no_grad() 안에서 self.q(torch.tensor(state)) 의 argmax 를 int 로
        if explore and random.random() < self.eps:
            return random.randrange(self.n_actions)      # 탐험: 아무 행동
        with torch.no_grad():
            return int(self.q(torch.tensor(state)).argmax())

    # ── ①에 저장 ────────────────────────────────────────────────
    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    # ── ③④ 학습 한 스텝 ────────────────────────────────────────
    def train_step(self):
        """버퍼에서 무작위 배치를 뽑아 TD 목표로 4박자 학습. loss 반환 (배치 부족하면 None)."""
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        # 내가 그 행동에 대해 예측했던 Q값: Q(s)[a]
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # TODO 3. TD 목표를 타겟 계산기로 만든다 (③):
        #   with torch.no_grad():                       ← 목표에는 기울기 금지
        #       q_next = self.q_target(s2).max(1).values
        #       target = r + self.gamma * q_next * (1 - d)   ← done 이면 미래항 제거
        with torch.no_grad():
            if self.double:                                   # M2: 선택은 온라인, 평가는 타겟
                next_a = self.q(s2).argmax(1, keepdim=True)
                q_next = self.q_target(s2).gather(1, next_a).squeeze(1)
            else:                                             # M1: 기존 그대로
                q_next = self.q_target(s2).max(1).values
            target = r + self.gamma * q_next * (1 - d)     # 목표 = 오늘 보상 + 0.99×내일의 최선

        # TODO 4. 4박자 (06 과 동일: loss → zero_grad → backward → step)
        #   loss = F.smooth_l1_loss(q_sa, target)
        loss = F.smooth_l1_loss(q_sa, target)    # 내 예측(q_sa) vs 방금 만든 목표
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타겟 계산기 주기적 갱신 + ε 감쇠
        self.step_count += 1
        if self.step_count % self.target_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return float(loss.detach())
