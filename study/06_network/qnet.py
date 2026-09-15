"""06단계: Q네트워크와 학습 4박자. lesson.md 를 먼저 읽는다.

채점:  python study/06_network/check_06.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_qnet(state_dim: int, hidden: int = 64, n_actions: int = 3) -> nn.Module:
    """state_dim 개를 받아 Q값 n_actions 개를 내는 MLP.

    구조: Linear(state_dim→hidden) → ReLU → Linear(hidden→hidden) → ReLU → Linear(hidden→n_actions)
    ★ 마지막 층 뒤에는 ReLU 를 붙이지 않는다 (Q값은 음수 가능).
    """
    # TODO 1.  힌트: nn.Sequential(nn.Linear(...), nn.ReLU(), ...)
    return nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))


def train_step(model: nn.Module, optimizer, states: torch.Tensor,
               target_q: torch.Tensor) -> float:
    """학습 4박자 한 번: forward → loss → backward → step. loss 값을 반환.

    states:   (배치, state_dim)
    target_q: (배치, n_actions)  — 맞춰야 할 정답 Q값들
    """
    # TODO 2. 4박자를 순서대로:
    #   pred = model(states)
    #   loss = F.smooth_l1_loss(pred, target_q)
    #   optimizer.zero_grad() → loss.backward() → optimizer.step()
    pred = model(states)
    loss = F.smooth_l1_loss(pred,target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fit_toy(seed: int = 0, steps: int = 300, state_dim: int = 20):
    """장난감 학습: 정답 규칙을 흉내 내게 해서 loss 가 내려가는지 본다.

    정답 규칙(우리가 임의로 정한 것): 첫 번째 Q = state 평균, 두 번째 = 최댓값, 세 번째 = 0.
    반환: (첫 loss, 마지막 loss, model)
    """
    torch.manual_seed(seed)
    model = build_qnet(state_dim)
    # TODO 3. Adam 옵티마이저를 만들어라 (학습률 lr=1e-3)
    #   힌트: torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr =1e-3 )

    losses = []
    for _ in range(steps):
        states = torch.randn(64, state_dim)                     # 랜덤 state 배치
        target_q = torch.stack([states.mean(dim=1),             # 정답 Q 3개
                                states.max(dim=1).values,
                                torch.zeros(64)], dim=1)
        losses.append(train_step(model, optimizer, states, target_q))
    return losses[0], losses[-1], model


if __name__ == "__main__":
    first, last, _ = fit_toy()
    print(f"첫 loss {first:.4f} → 마지막 loss {last:.4f}")
    print("내려갔으면 4박자가 도는 것. check_06 으로 정식 채점하세요.")
