"""08단계: 실험 실행기. 빈칸 없음 — lesson 의 Double DQN 패치(dqn.py) 후 실행.

  python study/08_runner/run_experiment.py --quick     # 배관 점검
  python study/08_runner/run_experiment.py             # 본실험 (밤에 걸어두기)

산출물: out/experiments.csv (전 실행 기록, 이어쓰기)
        out/summary.md      (모델·기준선 × 지표, 평균±표준편차 — 논문 표 초안)
        out/fig_<model>_<year>.png (시드 5개 자산 곡선)
"""
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ["02_split", "03_baselines", "04_env", "07_dqn"]:
    sys.path.insert(0, str(ROOT / "study" / p))
from split import make_folds  # noqa: E402
from metrics import cumulative_return, max_drawdown, sharpe, sortino  # noqa: E402
from baselines import buy_and_hold_pv, ma_crossover_pv, cash_pv, fixed_exposure_pv  # noqa: E402
from trading_env import TradingEnv  # noqa: E402
from dqn import DQNAgent  # noqa: E402

CSV = ROOT / "out" / "experiments.csv"
COLS = ["time", "code", "model", "test_year", "seed", "episodes",
        "ret", "excess_bh", "sharpe", "sortino", "mdd", "trades"]


def metrics_row(model, year, seed, episodes, pv, trades, bh_ret, code="005930"):
    return {"time": datetime.now().strftime("%m-%d %H:%M"), "code": code, "model": model,
            "test_year": year, "seed": seed, "episodes": episodes,
            "ret": round(cumulative_return(pv), 4),
            "excess_bh": round(cumulative_return(pv) - bh_ret, 4),
            "sharpe": round(sharpe(pv), 3), "sortino": round(sortino(pv), 3),
            "mdd": round(max_drawdown(pv), 4), "trades": trades}


def append_rows(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(CSV, mode="a", header=not CSV.exists(), index=False)


def run_agent(train_df, test_df, seed, episodes, double):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = TradingEnv(train_df)
    agent = DQNAgent(state_dim=len(env.reset()), double=double)
    for _ in range(episodes):                       # 학습
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2
    test_env = TradingEnv(test_df)                  # 테스트: ε 무시, 학습 없음
    s, done = test_env.reset(), False
    pv = [test_env.asset]
    while not done:
        s, _, done = test_env.step(agent.act(s, explore=False))
        pv.append(test_env.asset)
    return pd.Series(pv, index=test_df.index[test_env.window - 1:test_env.t]), test_env.trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dqn", "double"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--code", default="005930")
    ap.add_argument("--quick", action="store_true", help="fold 1 × seed 1 × ep 2 로 배관 점검")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.years, args.episodes = [1], [2024], 2

    feat = pd.read_csv(ROOT / "out" / f"{args.code}_features.csv",
                       parse_dates=["date"], index_col="date")
    folds = {te.index[0].year: (tr, te) for tr, te in make_folds(feat, args.years)}

    all_rows = []
    for year, (train_df, test_df) in folds.items():
        close = test_df["close"]
        bh = buy_and_hold_pv(close); bh_ret = cumulative_return(bh)
        ma_pv, ma_tr = ma_crossover_pv(close)
        base = [metrics_row("buy_hold", year, 0, 0, bh, None, bh_ret, args.code),
                metrics_row("ma_cross", year, 0, 0, ma_pv, ma_tr, bh_ret, args.code),
                metrics_row("cash", year, 0, 0, cash_pv(close), 0, bh_ret, args.code),
                metrics_row("fixed50", year, 0, 0, fixed_exposure_pv(close), None, bh_ret, args.code)]
        append_rows(base); all_rows += base
        print(f"[{year}] 기준선 4종 기록 (B&H {bh_ret:+.2%})")

        for model in args.models:
            curves = {}
            for seed in args.seeds:
                pv, trades = run_agent(train_df, test_df, seed, args.episodes,
                                       double=(model == "double"))
                row = metrics_row(model, year, seed, args.episodes, pv, trades, bh_ret, args.code)
                append_rows([row]); all_rows.append(row)
                curves[seed] = pv
                print(f"[{year}] {model} seed {seed}: {row['ret']:+.2%} "
                      f"(초과 {row['excess_bh']:+.2%}, 거래 {row['trades']})")
            try:                                    # 시드별 자산 곡선 그림
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                for sd, pv in curves.items():
                    ax.plot(pv.index, pv.values / pv.iloc[0], label=f"seed {sd}", alpha=0.8)
                ax.plot(bh.index, bh.values / bh.iloc[0], "k--", label="Buy&Hold")
                ax.set_title(f"{model} — test {year} (episodes {args.episodes})")
                ax.legend(fontsize=8); fig.tight_layout()
                fig.savefig(ROOT / "out" / f"fig_{args.code}_{model}_{year}.png", dpi=120)
                plt.close(fig)
            except Exception as e:  # noqa: BLE001
                print(f"  (그림 생략: {e})")

    # ── 논문 표 초안: 모델별 평균 ± 표준편차 ──
    df = pd.DataFrame(all_rows)
    lines = ["# 실험 요약 (이번 실행분)", "",
             f"folds {args.years} × seeds {args.seeds} × episodes {args.episodes}", "",
             "| 전략 | 수익률 | B&H 대비 초과 | Sortino | MDD | 거래 |",
             "|---|---|---|---|---|---|"]
    for model, g in df.groupby("model"):
        def ms(col, pct=True):
            m, s_ = g[col].mean(), g[col].std()
            return (f"{m:+.2%} ± {s_:.2%}" if pct else f"{m:.2f} ± {s_:.2f}") if len(g) > 1 \
                else (f"{m:+.2%}" if pct else f"{m:.2f}")
        lines.append(f"| {model} | {ms('ret')} | {ms('excess_bh')} | "
                     f"{ms('sortino', pct=False)} | {ms('mdd')} | {g['trades'].mean():.0f} |")
    out_md = ROOT / "out" / "summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n요약: {out_md}\n전체 기록: {CSV}")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
