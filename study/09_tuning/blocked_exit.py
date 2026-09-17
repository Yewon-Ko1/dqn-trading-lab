"""09단계 (측정): 최소보유 제약에 막힌 매도가 몇 번이고, 막힌 뒤 주가가 어떻게 됐나. 빈칸 없음.

왜 만들었나
  `min_hold=10` 은 매수 후 10거래일간 SELL 을 HOLD 로 바꾼다(액션 마스킹). 그래서 급락 국면에서
  에이전트가 나가려 해도 못 나간다. "일정 급락이면 예외를 허용해야 하지 않나" 라는 물음에
  **새 하이퍼파라미터(임계값)를 도입하지 않고** 답하기 위한 측정이다.

  D-03.5b·D-09 를 마지막으로 튜닝을 동결하기로 사전 등록했으므로, 이것은 새 실험이 아니라
  확정 설정의 동작을 관찰하는 진단이다. 채택 규칙도 없고 설정도 바꾸지 않는다.

무엇을 재나 (막힌 매도 1건 = 1행)
  dd_entry : 막힌 시점의 진입가 대비 손익 (-0.07 이면 7% 물려 있었다는 뜻)
  fwd5/fwd10 : 막힌 날부터 5·10 거래일 뒤까지의 주가 수익률
               → 음수면 "그때 나갔어야 했다"(예외가 이득), 양수면 "참길 잘했다"(예외가 손해)
  to_exit  : 막힌 날부터 실제로 포지션을 청산한 날까지의 주가 수익률 (실현된 결과)
  hold_days: 막힌 시점의 보유 일수 (제약 10일 중 며칠째였나)

읽는 법
  fwd10 평균이 뚜렷한 음수이고 dd_entry 가 깊은 건이 많다  → 손절 예외가 실제로 이득이었을 것
  fwd10 평균이 0 근처이거나 양수                          → 예외를 뒀어도 이득이 없거나 손해
  막힌 횟수 자체가 적다                                    → 예외를 둬도 영향이 미미

실행:  python study/09_tuning/blocked_exit.py            # 확정 설정, 2018·2019·2020, 시드 1~20
       python study/09_tuning/blocked_exit.py --seeds 1 2   # 빠른 확인
산출물: out/blocked_exit.csv (막힌 매도 1건당 1행) + 끝에 요약표
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
for p in ["02_split", "03_baselines", "04_env", "07_dqn", "09_tuning"]:
    sys.path.insert(0, str(ROOT / "study" / p))
from split import make_folds            # noqa: E402
from metrics import cumulative_return   # noqa: E402
from baselines import buy_and_hold_pv   # noqa: E402
from trading_env import TradingEnv      # noqa: E402
from dqn import DQNAgent                # noqa: E402
from tune import FSETS, CHURN, with_warmup, add_extra_features   # noqa: E402
from diag import CONFIRMED              # noqa: E402

OUT = ROOT / "out"
CSV = OUT / "blocked_exit.csv"
COLS = ["time", "code", "test_year", "seed", "date", "hold_days", "dd_entry",
        "fwd5", "fwd10", "to_exit", "n_blocked", "n_sell_req", "n_steps"]


def run_one(train_df, test_df, seed, cfg, feat):
    """확정 설정으로 학습한 뒤 평가 구간을 돌면서, 제약에 막힌 SELL 을 기록한다."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fcols = FSETS[cfg["features"]]
    extra_state, min_hold = CHURN[cfg["churn"]]
    env = TradingEnv(train_df, window=cfg["window"], features=fcols, extra_state=extra_state,
                     min_hold=min_hold, trade_penalty=cfg["penalty"])
    agent = DQNAgent(state_dim=len(env.reset()), gamma=cfg["gamma"], lr=cfg["lr"], hidden=cfg["hidden"],
                     target_every=cfg["target_every"], eps_decay=cfg["eps_decay"],
                     n_step=cfg.get("n_step", 1))
    for _ in range(cfg["episodes"]):
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2

    test_ext = with_warmup(feat, test_df, cfg["window"])
    te = TradingEnv(test_ext, window=cfg["window"], features=fcols,
                    extra_state=extra_state, min_hold=min_hold)
    s, done = te.reset(), False
    held = [te.shares > 0]
    blocked, n_sell_req, i = [], 0, 0        # i = 이번 스텝이 집행되는 시점의 인덱스
    while not done:
        a = agent.act(s, explore=False)
        if a == te.SELL and te.shares > 0:
            n_sell_req += 1
            if te.hold_days < te.min_hold:   # 제약에 막힘 (env 가 HOLD 로 바꿔 집행한다)
                blocked.append((i, te.hold_days, te.entry_price))
        s, _, done = te.step(a)
        held.append(te.shares > 0)
        i += 1
    idx = test_ext.index[te.window: te.t + 1]
    price = test_ext["close"].reindex(idx).to_numpy(dtype=float)
    assert idx[0] == test_df.index[0] and len(idx) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    held = np.array(held, dtype=bool)
    return blocked, price, idx, held, n_sell_req, len(idx)


def events(blocked, price, idx, held, n_sell_req, n_steps, year, seed, code):
    rows = []
    for i, hd, entry in blocked:
        def fwd(k):
            j = i + k
            return round(price[j] / price[i] - 1, 4) if j < len(price) else np.nan
        j = next((k for k in range(i + 1, len(held)) if not held[k]), None)   # 실제 청산 시점
        rows.append({"time": datetime.now().strftime("%m-%d %H:%M"), "code": code,
                     "test_year": year, "seed": seed, "date": idx[i].date(),
                     "hold_days": hd,
                     "dd_entry": round(price[i] / entry - 1, 4) if entry else np.nan,
                     "fwd5": fwd(5), "fwd10": fwd(10),
                     "to_exit": round(price[j] / price[i] - 1, 4) if j is not None else np.nan,
                     "n_blocked": len(blocked), "n_sell_req": n_sell_req, "n_steps": n_steps})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    args = ap.parse_args()

    cfg = CONFIRMED
    feat = pd.read_csv(OUT / f"{args.code}_features.csv", parse_dates=["date"], index_col="date").sort_index()
    feat = add_extra_features(feat)          # tune.py·diag.py 와 동일한 전처리
    print(f"설정 {cfg}\n폴드 {args.years} | 시드 {args.seeds[0]}~{args.seeds[-1]} | "
          f"min_hold={CHURN[cfg['churn']][1]}일\n")

    rows, per_run = [], []
    for train_df, test_df in make_folds(feat, args.years, train_len=cfg["train_len"]):
        year = test_df.index[0].year
        for seed in args.seeds:
            b, price, idx, held, nsr, nst = run_one(train_df, test_df, seed, cfg, feat)
            ev = events(b, price, idx, held, nsr, nst, year, seed, args.code)
            if ev:
                pd.DataFrame(ev, columns=COLS).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
            rows += ev
            per_run.append({"test_year": year, "seed": seed, "n_blocked": len(b),
                            "n_sell_req": nsr, "n_steps": nst})
            print(f"  [{year}] seed {seed:>2}: 매도 시도 {nsr:>3}회 중 제약에 막힘 {len(b):>3}회")

    pr = pd.DataFrame(per_run)
    df = pd.DataFrame(rows)
    print(f"\n── 막힌 매도 건수 ──")
    print(pr.groupby("test_year")[["n_sell_req", "n_blocked"]].mean().round(1).to_string())
    print(f"  전체: 매도 시도 {pr['n_sell_req'].mean():.1f}회/실행 중 {pr['n_blocked'].mean():.1f}회 막힘 "
          f"({pr['n_blocked'].sum() / max(pr['n_sell_req'].sum(), 1):.0%})")
    if df.empty:
        print("\n막힌 매도가 없음 → 예외 규칙은 아무 영향이 없다."); return

    print(f"\n── 막힌 뒤 주가 (막힌 건 {len(df)}개) ──")
    print(df.groupby("test_year")[["dd_entry", "fwd5", "fwd10", "to_exit", "hold_days"]].mean().round(4).to_string())
    m5, m10, mex = df["fwd5"].mean(), df["fwd10"].mean(), df["to_exit"].mean()
    print(f"  전체 평균: dd_entry {df['dd_entry'].mean():+.2%} | "
          f"fwd5 {m5:+.2%} | fwd10 {m10:+.2%} | to_exit {mex:+.2%}")

    print(f"\n── 손절 임계값별: 그 수준까지 물린 채 막힌 건들 ──")
    print(f"  {'임계':>6} {'해당 건수':>8} {'비중':>6} {'이후 fwd10':>10} {'실현 to_exit':>12}")
    for th in [-0.03, -0.05, -0.07, -0.10]:
        sub = df[df["dd_entry"] <= th]
        if len(sub) == 0:
            print(f"  {th:>6.0%} {0:>8} {'—':>6} {'—':>10} {'—':>12}"); continue
        print(f"  {th:>6.0%} {len(sub):>8} {len(sub)/len(df):>6.0%} "
              f"{sub['fwd10'].mean():>+10.2%} {sub['to_exit'].mean():>+12.2%}")
    print(f"\n→ {CSV}")
    print("\n읽는 법: fwd10 이 뚜렷한 음수면 그때 나갔어야 했다는 뜻(예외가 이득).")
    print("         0 근처이거나 양수면 예외를 뒀어도 이득이 없거나 오히려 손해였다.")


if __name__ == "__main__":
    main()
