"""07단계 마무리: 진짜 데이터로 첫 DQN 학습. 빈칸 없음 — check_07 통과 후 실행만.

  python study/07_dqn/train.py                     # 삼성전자, 시드 1, 에피소드 10
  python study/07_dqn/train.py --seed 3 --episodes 15

학습 구간 2020~2023, 테스트 2024 (02단계 분할 그대로).
끝나면 out/train_seed{N}.txt 에 결과를 남긴다 — 시드별로 돌려서 비교해 볼 것.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "study" / "04_env"))
sys.path.insert(0, str(ROOT / "study" / "07_dqn"))
from trading_env import FEATURES, TradingEnv  # noqa: E402
from dqn import DQNAgent  # noqa: E402


def run_test(env, agent):
    """테스트: ε 무시(explore=False), 학습 없음."""
    s = env.reset()
    done = False
    while not done:
        s, _, done = env.step(agent.act(s, explore=False))
    return env.asset, env.trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="학습 후 저장할 PyTorch checkpoint 경로. 예: models/dqn_policy.pt",
    )
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    feat = pd.read_csv(ROOT / "out" / "005930_features.csv",
                       parse_dates=["date"], index_col="date")
    train_df = feat.loc["2020":"2023"]
    test_df = feat.loc["2024":"2024"]

    env = TradingEnv(train_df)
    first_state = env.reset()
    state_dim = len(first_state)
    agent = DQNAgent(state_dim=state_dim)

    print(f"seed {args.seed} | 학습 {len(train_df)}일(2020~2023) → 테스트 {len(test_df)}일(2024)")
    for ep in range(1, args.episodes + 1):
        s = env.reset()
        total_r, done = 0.0, False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()                     # ← rltrader 와 달리 매 스텝
            s, total_r = s2, total_r + r
        print(f"  ep {ep:2d}/{args.episodes}  누적보상 {total_r:+.4f}  "
              f"학습구간 최종자산 {env.asset:,.0f}  ε={agent.eps:.3f}  거래 {env.trades}")

    test_env = TradingEnv(test_df)
    asset, trades = run_test(test_env, agent)
    model_ret = asset / 10_000_000 - 1
    bh_ret = test_df["close"].iloc[-1] / test_df["close"].iloc[test_env.window] - 1
    line = (f"seed {args.seed} | 2024 테스트: 모델 {model_ret:+.2%} (거래 {trades}회) "
            f"vs Buy&Hold {bh_ret:+.2%} vs 현금 0.00%")
    print("\n" + line)
    out = ROOT / "out" / f"train_seed{args.seed}.txt"
    out.write_text(line + "\n", encoding="utf-8")
    print(f"기록: {out}")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": agent.q.state_dict(),
                "state_dim": state_dim,
                "hidden": 64,
                "window": env.window,
                "features": FEATURES,
                "actions": ["buy", "sell", "hold"],
                "seed": args.seed,
                "episodes": args.episodes,
                "train_range": "2020-2023",
                "test_range": "2024",
            },
            ckpt,
        )
        print(f"모델 checkpoint: {ckpt}")
    print("\n※ 한 시드는 우연이다. --seed 1~5 로 다섯 번 돌려 평균±표준편차로 보는 것까지가 07의 끝.")


if __name__ == "__main__":
    main()
