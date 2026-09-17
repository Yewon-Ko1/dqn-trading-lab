"""09단계 (진단): 확정 설정이 '언제 들고 있었는가'를 기록한다. 빈칸 없음.

왜 따로 만들었나
  out/tuning.csv 에는 ret·excess_bh·sortino·mdd·trades 만 있다. 이 다섯 개로는
  "상승장에서 현금을 들고 있었다" 와 "샀는데 일찍 팔았다" 를 구분할 수 없다.
  보완 시그널을 고르려면 그 구분이 먼저 필요하다. 정책 가중치를 저장하지 않으므로
  확정 설정으로 검증 폴드만 다시 학습해 궤적을 남긴다.

무엇을 건드리지 않나
  tune.py·trading_env.py·tuning.csv 는 읽기만 한다. 자동 체인과 동시에 돌려도
  기록이 섞이지 않도록 산출물은 out/diag.csv 로 분리했다. (단 CPU 는 나눠 쓰게 되므로
  체인이 끝난 뒤 돌리는 편이 빠르다.)

국면 정의 (사전 등록 대상)
  bull_t = 1 if (KOSPI_t-1 / MA60(KOSPI)_t-1 - 1) > 0 else 0
  의사결정 시점에 확정된 값만 쓰도록 하루 밀어서(shift) 붙인다. 진단용 렌즈이자
  D-03.10 게이트 후보의 신호 정의이기도 하므로, 이 정의를 바꾸려면 실험 전에 기록해야 한다.

실행:
  python study/09_tuning/diag.py                      # 확정 설정, 2018·2019·2020, 시드 1~20
  python study/09_tuning/diag.py --double             # Double DQN 으로 같은 진단
  python study/09_tuning/diag.py --seeds 1 2 3        # 빠른 확인
  python study/09_tuning/diag.py --years 2017 --train-len 3   # 미사용 폴드 (확인용)

산출물: out/diag.csv (실행당 1행) + 끝에 요약표.
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
from metrics import cumulative_return, max_drawdown, sortino   # noqa: E402
from baselines import buy_and_hold_pv   # noqa: E402
from trading_env import TradingEnv      # noqa: E402
from dqn import DQNAgent                # noqa: E402
from tune import FSETS, CHURN, with_warmup, add_extra_features   # noqa: E402

OUT = ROOT / "out"
CSV = OUT / "diag.csv"
COLS = ["time", "code", "algo", "test_year", "seed", "ret", "bh_ret", "excess_bh", "sortino", "mdd",
        "trades", "hold_ratio", "hold_bull", "hold_bear", "avg_hold_days", "n_spells",
        "capture_up", "capture_dn", "day_up", "day_dn", "entry_lag", "bull_days"]

# 자동 체인이 확정한 설정 (out/auto_chain.md 최종 줄)
CONFIRMED = dict(episodes=10, window=10, features="slim5", hidden=64, lr=1e-3, gamma=0.99,
                 target_every=500, eps_decay=0.999, train_len=5, churn="hold10", penalty=0.0)


def bull_series(index: pd.DatetimeIndex, ma: int = 60) -> pd.Series:
    """KOSPI 가 MA60 위면 1. shift(1) 로 '전날까지 확정된 값' 만 쓴다."""
    k = pd.read_csv(OUT / "kospi.csv", parse_dates=["date"], index_col="date")["kospi"].sort_index()
    trend = k / k.rolling(ma).mean() - 1
    return (trend.shift(1) > 0).reindex(index).astype("boolean")


def spells(held: np.ndarray):
    """보유 구간들의 길이 목록. [F,T,T,F,T] -> [2, 1]"""
    out, run = [], 0
    for h in held:
        if h:
            run += 1
        elif run:
            out.append(run); run = 0
    if run:
        out.append(run)
    return out


def entry_lag(held: np.ndarray, bull: np.ndarray) -> float:
    """상승 국면 전환(비상승→상승) 후 실제로 보유하기까지 걸린 거래일수의 평균.
    해당 상승 국면이 끝날 때까지 한 번도 안 샀으면 그 국면 길이를 벌점으로 쓴다."""
    lags = []
    for i in range(1, len(bull)):
        if not (bull[i] and not bull[i - 1]):
            continue
        j = i
        while j < len(bull) and bull[j]:
            if held[j]:
                lags.append(j - i); break
            j += 1
        else:
            lags.append(j - i)
    return float(np.mean(lags)) if lags else float("nan")


def run_one(train_df, test_df, seed, cfg, feat, double: bool):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fcols = FSETS[cfg["features"]]
    extra_state, min_hold = CHURN[cfg["churn"]]
    env = TradingEnv(train_df, window=cfg["window"], features=fcols, extra_state=extra_state,
                     min_hold=min_hold, trade_penalty=cfg["penalty"])
    agent = DQNAgent(state_dim=len(env.reset()), gamma=cfg["gamma"], lr=cfg["lr"], hidden=cfg["hidden"],
                     target_every=cfg["target_every"], eps_decay=cfg["eps_decay"], double=double)
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
    pv, held = [te.asset], [te.shares > 0]        # t=window 시점(테스트 연도 첫날)의 상태
    while not done:
        s, _, done = te.step(agent.act(s, explore=False))
        pv.append(te.asset); held.append(te.shares > 0)
    idx = test_ext.index[te.window: te.t + 1]
    pv = pd.Series(pv, index=idx)
    assert pv.index[0] == test_df.index[0] and len(pv) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    return pv, np.array(held, dtype=bool), te.trades, test_ext["close"].reindex(idx)


def diagnose(pv, held, trades, price, bull_all) -> dict:
    ret = price.pct_change().to_numpy()            # ret[i] = i-1 → i 구간 수익률
    pos = held[:-1]                                # 그 구간을 보유한 상태는 직전 시점의 포지션
    r = ret[1:]
    bull = bull_all.reindex(pv.index).fillna(False).to_numpy(dtype=bool)
    up, dn = r > 0, r < 0
    sp = spells(held)
    return {
        "ret": round(cumulative_return(pv), 4),
        "sortino": round(sortino(pv), 3),
        "mdd": round(max_drawdown(pv), 4),
        "trades": trades,
        "hold_ratio": round(float(held.mean()), 3),
        "hold_bull": round(float(held[bull].mean()), 3) if bull.any() else float("nan"),
        "hold_bear": round(float(held[~bull].mean()), 3) if (~bull).any() else float("nan"),
        "avg_hold_days": round(float(np.mean(sp)), 1) if sp else 0.0,
        "n_spells": len(sp),
        # 수익률 가중 포착률: 오른 날의 상승분 중 실제로 먹은 비율 (하락은 낮을수록 좋다)
        "capture_up": round(float(r[up][pos[up]].sum() / r[up].sum()), 3) if up.any() else float("nan"),
        "capture_dn": round(float(r[dn][pos[dn]].sum() / r[dn].sum()), 3) if dn.any() else float("nan"),
        "day_up": round(float(pos[up].mean()), 3) if up.any() else float("nan"),
        "day_dn": round(float(pos[dn].mean()), 3) if dn.any() else float("nan"),
        "entry_lag": round(entry_lag(held, bull), 1),
        "bull_days": round(float(bull.mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    ap.add_argument("--double", action="store_true", help="Double DQN 으로 진단 (같은 설정·같은 시드)")
    ap.add_argument("--train-len", type=int, default=CONFIRMED["train_len"])
    ap.add_argument("--ma", type=int, default=60, help="국면 판정 이동평균 (사전 등록: 60)")
    args = ap.parse_args()

    cfg = dict(CONFIRMED, train_len=args.train_len)
    algo = "double" if args.double else "dqn"
    feat = pd.read_csv(OUT / f"{args.code}_features.csv", parse_dates=["date"], index_col="date").sort_index()
    feat = add_extra_features(feat)      # tune.py 와 동일한 전처리 — 빠뜨리면 학습 구간이 어긋난다
    bull_all = bull_series(feat.index, args.ma)
    print(f"설정 {cfg}\n알고리즘 {algo} | 폴드 {args.years} | 시드 {args.seeds[0]}~{args.seeds[-1]} | 국면 MA{args.ma}\n")

    rows = []
    for train_df, test_df in make_folds(feat, args.years, train_len=cfg["train_len"]):
        year = test_df.index[0].year
        bh = cumulative_return(buy_and_hold_pv(test_df["close"]))
        for seed in args.seeds:
            pv, held, trades, price = run_one(train_df, test_df, seed, cfg, feat, args.double)
            d = diagnose(pv, held, trades, price, bull_all)
            row = {"time": datetime.now().strftime("%m-%d %H:%M"), "code": args.code, "algo": algo,
                   "test_year": year, "seed": seed, "bh_ret": round(bh, 4),
                   "excess_bh": round(d["ret"] - bh, 4), **d}
            pd.DataFrame([row], columns=COLS).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
            rows.append(row)
            print(f"  [{year}] seed {seed:>2}: 초과 {row['excess_bh']:+.1%} | 보유 {d['hold_ratio']:.0%} "
                  f"(상승국면 {d['hold_bull']:.0%} / 비상승 {d['hold_bear']:.0%}) | "
                  f"상승포착 {d['capture_up']:.0%} 하락노출 {d['capture_dn']:.0%} | "
                  f"보유기간 {d['avg_hold_days']:.0f}일×{d['n_spells']}회 | 진입지연 {d['entry_lag']:.0f}일")

    df = pd.DataFrame(rows)
    num = ["excess_bh", "hold_ratio", "hold_bull", "hold_bear", "capture_up", "capture_dn",
           "avg_hold_days", "n_spells", "entry_lag", "trades"]
    print(f"\n── {algo} 연도별 평균 (시드 {len(args.seeds)}개) ──")
    print(df.groupby("test_year")[num].mean().round(3).to_string())
    print(f"\n전체 평균 초과 {df['excess_bh'].mean():+.1%} | "
          f"상승국면 보유 {df['hold_bull'].mean():.0%} vs 비상승 보유 {df['hold_bear'].mean():.0%}")
    print(f"→ {CSV}")
    print("\n읽는 법: 상승국면 보유가 비상승보다 낮으면 '국면을 거꾸로 타고 있다'는 뜻 → 추세 게이트 후보.")
    print("         상승포착이 낮은데 보유기간이 짧으면 '사긴 사는데 일찍 판다' → n-step 또는 보유 제약.")
    print("         진입지연이 길면 '상승 전환을 늦게 따라간다' → 야간갭·시장 모멘텀 후보.")


if __name__ == "__main__":
    main()
