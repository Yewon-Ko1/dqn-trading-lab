"""KAIRO policy adapter for checkpoints exported from dqn-trading-lab.

Expected use inside the KAIRO backend:

    from app.agent.dqn_lab_policy import register_dqn_lab_policy
    register_dqn_lab_policy("dqn-lab-v1", "models/dqn_policy.pt")

The adapter mirrors the price/volume feature layout used by
study/04_env/trading_env.py: 20 days * 8 features. The final 3 portfolio
values use the closest fields available in KAIRO's live Observation.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from app.agent.policy import Observation, Policy, PolicyOutput, register_policy


FEATURE_KEYS = [
    "ret1",
    "ret5",
    "ret20",
    "close_ma5_ratio",
    "close_ma20_ratio",
    "ma20_ma60_ratio",
    "vol20",
    "volume_ma20_ratio",
]

ACTION_NAMES = ["buy", "sell", "hold"]


def build_qnet(state_dim: int, hidden: int = 64, n_actions: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_sample(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator - 1 if denominator else 0.0


def _feature_row(closes: list[float], volumes: list[float], i: int) -> list[float]:
    ma5 = _mean(closes[i - 4 : i + 1])
    ma20 = _mean(closes[i - 19 : i + 1])
    ma60 = _mean(closes[i - 59 : i + 1])
    ret1_window = [_ratio(closes[j], closes[j - 1]) for j in range(i - 19, i + 1)]

    return [
        _ratio(closes[i], closes[i - 1]),
        _ratio(closes[i], closes[i - 5]),
        _ratio(closes[i], closes[i - 20]),
        _ratio(closes[i], ma5),
        _ratio(closes[i], ma20),
        _ratio(ma20, ma60),
        _std_sample(ret1_window),
        _ratio(volumes[i], _mean(volumes[i - 19 : i + 1])),
    ]


def build_state_vector(obs: Observation, window: int = 20) -> list[float]:
    """Build the DQN state from KAIRO's live Observation."""
    min_candles = window + 59
    if len(obs.candles) < min_candles:
        raise ValueError(f"need at least {min_candles} candles, got {len(obs.candles)}")

    candles = obs.candles
    closes = [float(c.close) for c in candles]
    volumes = [float(c.volume) for c in candles]
    start = len(candles) - window

    features: list[float] = []
    for i in range(start, len(candles)):
        features.extend(_feature_row(closes, volumes, i))

    portfolio_state = [
        float(obs.position_ratio),
        float(obs.unrealized_pct) / 100.0,
        float(obs.cash_ratio),
    ]
    return features + portfolio_state


class DQNLabPolicy(Policy):
    name = "dqn-lab-v1"

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.path = Path(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(self.path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        self.window = int(checkpoint.get("window", 20))
        self.state_dim = int(checkpoint.get("state_dim", self.window * len(FEATURE_KEYS) + 3))
        hidden = int(checkpoint.get("hidden", 64))
        self.net = build_qnet(self.state_dim, hidden=hidden, n_actions=len(ACTION_NAMES))
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()

    def act(self, obs: Observation) -> PolicyOutput:
        try:
            state = build_state_vector(obs, self.window)
        except ValueError as exc:
            return PolicyOutput(
                action="hold",
                confidence=0.0,
                q_values={"buy": 0.0, "hold": 1.0, "sell": 0.0},
                reason=f"DQN state unavailable: {exc}",
            )

        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_tensor = self.net(x).squeeze(0)
            probs = torch.softmax(q_tensor, dim=0)
            action_idx = int(torch.argmax(q_tensor).item())

        q_values = {
            action: float(q_tensor[idx].detach().cpu())
            for idx, action in enumerate(ACTION_NAMES)
        }
        confidence = float(probs[action_idx].detach().cpu())
        action = ACTION_NAMES[action_idx]
        return PolicyOutput(
            action=action,
            confidence=confidence,
            q_values=q_values,
            reason=f"DQN lab checkpoint {self.path.name}",
        )


def register_dqn_lab_policy(
    name: str = "dqn-lab-v1",
    checkpoint_path: str | Path = "models/dqn_policy.pt",
) -> bool:
    path = Path(checkpoint_path)
    if not path.exists():
        return False
    register_policy(name, DQNLabPolicy(path))
    return True
