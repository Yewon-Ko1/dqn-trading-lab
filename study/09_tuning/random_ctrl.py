"""09단계 (대조군): 학습된 정책이 무작위 정책보다 나은가. 빈칸 없음.

왜 필요한가
  diag.py 결과, 확정 설정의 에이전트는 시장 국면과 무관하게 약 73% 의 시간을 보유 상태로
  보냈다 (상승 국면 73.9% vs 비상승 72.2%, 차이가 표준오차 안쪽). 그런데 확정 설정에는
  min_hold=10 (산 뒤 10일간 매도 금지) 제약이 걸려 있고, 이 제약 자체가 구조적으로 보유
  비율을 끌어올린다. 따라서 "73% 보유"가 학습의 결과인지 제약의 결과인지 구분되지 않는다.

  무작위 행동에 같은 제약만 걸어 같은 폴드·같은 시드로 돌리면 그 구분이 된다.
    - 무작위가 비슷하면  → 학습이 기여한 것이 없다 (강한 음성 결과)
    - 무작위가 뚜렷이 나쁘면 → 학습은 작동했고, 국면 구분만 못 배운 것

  학습이 없으므로 신경망도 시드도 부동소수점 누적도 개입하지 않는다. 행동은 numpy PCG64
  정수 추출로만 정해지므로 기기가 달라도 같은 결과가 나온다 (decisions.md "실행 환경" 참조).

대조군
  rand_hold10 : 무작위 행동 + min_hold=10   (확정 설정과 같은 제약)
  rand_free   : 무작위 행동 + 제약 없음      (제약이 기여한 몫을 분리)

실행:  python study/09_tuning/random_ctrl.py            # 2018·2019·2020, 시드 1~20
산출물: out/random_ctrl.csv (diag.csv 와 같은 열 구성 — 바로 이어붙여 비교 가능)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ["02_split", "03_baselines", "04_env", "09_tuning"]:
    sys.path.insert(0, str(ROOT / "study" / p))
from split import make_folds                    # noqa: E402
from metrics import cumulative_return           # noqa: E402
from baselines import buy_and_hold_pv           # noqa: E402
from trading_env import TradingEnv              # noqa: E402
from tune import FSETS, with_warmup, add_extra_features   # noqa: E402
from diag import COLS, CONFIRMED, bull_series, diagnose   # noqa: E402

OUT = ROOT / "out"
CSV = OUT / "random_ctrl.csv"
CONTROLS = {"rand_hold10": 10, "rand_free": 0}


def run_one(test_df, seed, feat, min_hold, cfg):
    rng = np.random.default_rng(seed)
    fcols = FSETS[cfg["features"]]
    test_ext = with_warmup(feat, test_df, cfg["window"])
    te = TradingEnv(test_ext, window=cfg["window"], features=fcols, min_hold=min_hold)
    s, done = te.reset(), False
    pv, held = [te.asset], [te.shares > 0]
    while not done:
        _, _, done = te.step(int(rng.integers(0, 3)))
        pv.append(te.asset); held.append(te.shares > 0)
    idx = test_ext.index[te.window: te.t + 1]
    pv = pd.Series(pv, index=idx)
    assert pv.index[0] == test_df.index[0] and len(pv) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    return pv, np.array(held, dtype=bool), te.trades, test_ext["close"].reindex(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    ap.add_argument("--ma", type=int, default=60)
    args = ap.parse_args()

    cfg = CONFIRMED
    feat = pd.read_csv(OUT / f"{args.code}_features.csv", parse_dates=["date"], index_col="date").sort_index()
    feat = add_extra_features(feat)              # tune.py·diag.py 와 동일한 전처리
    bull_all = bull_series(feat.index, args.ma)
    print(f"대조군 {list(CONTROLS)} | 폴드 {args.years} | 시드 {args.seeds[0]}~{args.seeds[-1]}\n")

    rows = []
    for name, min_hold in CONTROLS.items():
        for train_df, test_df in make_folds(feat, args.years, train_len=cfg["train_len"]):
            year = test_df.index[0].year
            bh = cumulative_return(buy_and_hold_pv(test_df["close"]))
            for seed in args.seeds:
                pv, held, trades, price = run_one(test_df, seed, feat, min_hold, cfg)
                d = diagnose(pv, held, trades, price, bull_all)
                row = {"time": datetime.now().strftime("%m-%d %H:%M"), "code": args.code, "algo": name,
                       "test_year": year, "seed": seed, "bh_ret": round(bh, 4),
                       "excess_bh": round(d["ret"] - bh, 4), **d}
                pd.DataFrame([row], columns=COLS).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
                rows.append(row)
        s = [r for r in rows if r["algo"] == name]
        print(f"  {name:>12}: 초과 {np.mean([r['excess_bh'] for r in s]):+.1%} | "
              f"보유 {np.mean([r['hold_ratio'] for r in s]):.0%} "
              f"(상승국면 {np.nanmean([r['hold_bull'] for r in s]):.0%} / 비상승 {np.nanmean([r['hold_bear'] for r in s]):.0%}) | "
              f"거래 {np.mean([r['trades'] for r in s]):.0f}")

    df = pd.DataFrame(rows)
    num = ["excess_bh", "hold_ratio", "hold_bull", "hold_bear", "capture_up", "capture_dn",
           "avg_hold_days", "n_spells", "entry_lag", "trades"]
    print("\n── 대조군 × 연도 평균 ──")
    print(df.groupby(["algo", "test_year"])[num].mean().round(3).to_string())
    print(f"\n→ {CSV}")
    print("\n비교 대상: out/diag.csv (학습된 정책, 같은 폴드·같은 시드)")
    print("  학습 정책과 rand_hold10 의 초과수익률 차이가 표준오차 안쪽이면 → 학습 기여 없음.")


if __name__ == "__main__":
    main()
