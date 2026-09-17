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

    def push(self, s, a, r, s2, done, gpow=1.0):
        """gpow = 이 전이의 할인 계수 γ^n (1-step 이면 γ). n-step 전이를 담기 위해 함께 저장한다."""
        self.buf.append((s, a, r, s2, done, gpow))

    def sample(self, batch_size: int):
        """무작위 batch_size 개를 뽑아 텐서 5개로 돌려준다."""
        # TODO 1b. random.sample(self.buf, batch_size) 로 무작위 추출
        batch = random.sample(self.buf,batch_size )
        s, a, r, s2, d, g = zip(*batch)     # 튜플들을 항목별로 분리
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
                torch.tensor(g, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int = 3, gamma: float = 0.99,
                 lr: float = 1e-3, batch_size: int = 64, target_every: int = 500,
                 eps_start: float = 1.0, eps_min: float = 0.05, eps_decay: float = 0.999,
                 double: bool = False, hidden: int = 64, n_step: int = 1):
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
        # ── D-09: n-step return ────────────────────────────────────
        # 1-step TD 는 "길게 보유하면 좋다" 는 가치를 한 칸씩만 역전파한다. n-step 은 n 칸의 보상을
        # 먼저 합쳐 목표로 쓰므로 그 전파가 n 배 빠르다. n_step=1 이면 기존 동작과 완전히 동일하다.
        self.n_step = n_step
        self._nq = deque(maxlen=n_step)     # 아직 전이로 만들지 못한 최근 n 스텝

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
    def _emit(self):
        """_nq 전체를 하나의 n-step 전이로 만들어 버퍼에 넣는다 (_nq 자체는 건드리지 않는다).

        R = r_0 + γr_1 + ... + γ^(k-1)r_(k-1),  s' = k 스텝 뒤 상태,  할인 계수 = γ^k   (k = len(_nq))
        """
        R, g = 0.0, 1.0
        for _, _, r, _, _ in self._nq:
            R += g * r
            g *= self.gamma
        s0, a0 = self._nq[0][0], self._nq[0][1]
        s_last, done_last = self._nq[-1][3], self._nq[-1][4]
        self.buffer.push(s0, a0, R, s_last, done_last, g)

    def remember(self, s, a, r, s2, done):
        self._nq.append((s, a, r, s2, done))
        if len(self._nq) == self.n_step:
            self._emit()
        if done:                                  # 에피소드 끝: 꼬리(길이 n-1, n-2, ... 1)도 전이로
            if len(self._nq) < self.n_step:       # 에피소드가 n 보다 짧았던 경우
                self._emit()
            while len(self._nq) > 1:
                self._nq.popleft()
                self._emit()
            self._nq.clear()

    # ── ③④ 학습 한 스텝 ────────────────────────────────────────
    def train_step(self):
        """버퍼에서 무작위 배치를 뽑아 TD 목표로 4박자 학습. loss 반환 (배치 부족하면 None)."""
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d, g = self.buffer.sample(self.batch_size)

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
            target = r + g * q_next * (1 - d)   # 목표 = n 스텝 보상합 + γ^n × n 스텝 뒤의 최선

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
