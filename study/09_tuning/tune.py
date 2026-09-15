"""09단계: 하이퍼파라미터 튜닝 실행기. 빈칸 없음.

원칙 (decisions.md 참고):
  1. 튜닝은 검증 폴드(기본: 테스트 2018 하락·2019 상승·2020 V자)에서만 한다. 평가 연도 2021~2024는
     설정 확정 후 딱 한 번만 다시 돌린다 — 테스트 성적을 보고 고르는 순간 체리피킹이다.
  2. 한 번에 한 변수만 바꾼다. 모든 실행은 설정 전체와 함께 out/tuning.csv 에 기록된다.
  3. 보고는 시드 10개 평균 ± 표준편차 + 폴드별 순위. 동률 기준 ±3%p, 기본값 유지가 디폴트.

사용 예:
  python study/09_tuning/tune.py                              # 기준 설정, 검증 폴드
  python study/09_tuning/tune.py --episodes 20
  python study/09_tuning/tune.py --sweep episodes 10 20 40    # 한 변수 스윕
  python study/09_tuning/tune.py --sweep window 10 20 60
  python study/09_tuning/tune.py --window 10 --sweep features slim5 base8 plus10 structure market flow
  python study/09_tuning/tune.py --sweep hidden 32 64 128
  python study/09_tuning/tune.py --sweep lr 3e-4 1e-3 3e-3
  python study/09_tuning/tune.py --sweep gamma 0.95 0.99 0.995

산출물: out/tuning.csv (전 실행 기록), 실행 끝에 설정별 요약표 출력.
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
from metrics import cumulative_return, max_drawdown, sortino  # noqa: E402
from baselines import buy_and_hold_pv  # noqa: E402
from trading_env import TradingEnv, FEATURES  # noqa: E402
from dqn import DQNAgent  # noqa: E402

CSV = ROOT / "out" / "tuning.csv"
COLS = ["time", "code", "test_year", "seed",
        "episodes", "window", "features", "hidden", "lr", "gamma",
        "target_every", "eps_decay",
        "ret", "excess_bh", "sortino", "mdd", "trades"]

# ── 피처 세트 정의 (D-03 사전 등록, decisions.md 참고) ─────────────
# structure/market/market_vk/flow 컬럼은 study/01_features/extra_features.py 가 만든다.
FSETS = {
    "slim5": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio"],
    "base8": FEATURES,                          # 현행 8종 (기준)
    "plus10": FEATURES + ["rsi14", "macd_ratio"],                              # 기술지표 추가 (중복 계열)
    "structure": FEATURES + ["close_ma200_ratio", "close_hi252_ratio", "vol5_vol20", "hl_range"],  # 장기 국면·일중
    "market": FEATURES + ["kospi_ret5", "kospi_ret20", "rel20"],               # 시장 국면
    "market_vk": FEATURES + ["kospi_ret5", "kospi_ret20", "rel20", "vkospi_ma20_ratio"],  # + 공포지수 (vkospi.csv 필요)
    "flow": FEATURES + ["frgn5", "frgn20", "inst20"],                           # 외국인·기관 수급 (<code>_flow.csv 필요)
}


def add_extra_features(feat: pd.DataFrame) -> pd.DataFrame:
    """plus10 세트용 추가 지표 2종. 전부 과거 데이터만 사용 (룩어헤드 없음).

    rsi14      : 14일 RSI 를 0 중심으로 스케일 (rsi/100 - 0.5) — 다른 피처와 크기 통일
    macd_ratio : (EMA12 - EMA26) / close — 가격 무관한 비율형 MACD
    """
    out = feat.copy()
    close = out["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = (100 - 100 / (1 + rs)) / 100 - 0.5
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd_ratio"] = (ema12 - ema26) / close
    return out.dropna()


def with_warmup(feat: pd.DataFrame, test_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """테스트 구간 앞에 직전 n 거래일을 붙인다 (관측 창 워밍업).

    TradingEnv 는 첫 window 일을 관측 재료로만 쓰고 window 일째부터 거래한다. 워밍업 없이 테스트 연도만
    넣으면 에이전트는 1~3월을 건너뛴 채 시작하는데 B&H 는 1월 첫날부터 계산되어 비교가 어긋난다
    (예: 2019 삼성전자 연간 +44% vs 20일째부터 +20%). 직전 n 일은 과거 데이터라 룩어헤드가 아니다.
    """
    start = feat.index.get_loc(test_df.index[0])
    return feat.iloc[max(0, start - n): start + len(test_df)]


def run_one(train_df, test_df, seed, cfg, feat):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fcols = FSETS[cfg.features]
    env = TradingEnv(train_df, window=cfg.window, features=fcols)
    agent = DQNAgent(state_dim=len(env.reset()), gamma=cfg.gamma, lr=cfg.lr,
                     hidden=cfg.hidden, target_every=cfg.target_every,
                     eps_decay=cfg.eps_decay)
    for _ in range(cfg.episodes):
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2
    test_ext = with_warmup(feat, test_df, cfg.window)           # 워밍업 window 일 + 테스트 연도 전체
    test_env = TradingEnv(test_ext, window=cfg.window, features=fcols)
    s, done = test_env.reset(), False
    pv = [test_env.asset]                                        # t=window = 테스트 연도 첫 거래일
    while not done:
        s, _, done = test_env.step(agent.act(s, explore=False))
        pv.append(test_env.asset)
    pv = pd.Series(pv, index=test_ext.index[test_env.window:test_env.t + 1])
    assert pv.index[0] == test_df.index[0] and len(pv) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    return pv, test_env.trades


def run_config(feat, cfg, years, seeds, code):
    rows = []
    folds = {te.index[0].year: (tr, te) for tr, te in make_folds(feat, years)}
    for year, (train_df, test_df) in folds.items():
        bh_ret = cumulative_return(buy_and_hold_pv(test_df["close"]))
        for seed in seeds:
            pv, trades = run_one(train_df, test_df, seed, cfg, feat)
            ret = cumulative_return(pv)
            rows.append({"time": datetime.now().strftime("%m-%d %H:%M"), "code": code,
                         "test_year": year, "seed": seed,
                         "episodes": cfg.episodes, "window": cfg.window,
                         "features": cfg.features, "hidden": cfg.hidden,
                         "lr": cfg.lr, "gamma": cfg.gamma,
                         "target_every": cfg.target_every, "eps_decay": cfg.eps_decay,
                         "ret": round(ret, 4), "excess_bh": round(ret - bh_ret, 4),
                         "sortino": round(sortino(pv), 3),
                         "mdd": round(max_drawdown(pv), 4), "trades": trades})
            print(f"  [{year}] seed {seed}: {ret:+.2%} (초과 {ret - bh_ret:+.2%}, 거래 {trades})")
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    return df


def summarize(df, label):
    g = df.groupby("test_year")
    exc_m, exc_s = df["excess_bh"].mean(), df["excess_bh"].std()
    print(f"\n■ {label}")
    for y, gy in g:
        print(f"  {y}: 초과 {gy['excess_bh'].mean():+.2%} ± {gy['excess_bh'].std():.2%} "
              f"(승률 {(gy['excess_bh'] > 0).sum()}/{len(gy)})  MDD {gy['mdd'].mean():.2%}")
    print(f"  종합: 초과 {exc_m:+.2%} ± {exc_s:.2%}  |  시드 표준편차(연도 내 평균) "
          f"{g['ret'].std().mean():.2%}  |  거래 {df['trades'].mean():.0f}회")
    per_fold = {int(y): gy["excess_bh"].mean() for y, gy in g}
    seed_sd = g["ret"].std().mean()          # 연도 내 시드 std 의 평균 (폴드 간 차이는 제외)
    return exc_m, seed_sd, per_fold, df["trades"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020],
                    help="튜닝 폴드(하락·상승·V자). 평가 연도(2021~2024)는 확정 후에만!")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--features", choices=list(FSETS), default="base8")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-every", type=int, default=500)
    ap.add_argument("--eps-decay", type=float, default=0.999)
    ap.add_argument("--sweep", nargs="+", default=None,
                    metavar=("PARAM", "VALUES"),
                    help="예: --sweep episodes 10 20 40 (그 외 인자는 고정값으로 사용)")
    args = ap.parse_args()

    if any(y >= 2021 for y in args.years):
        print("⚠️  경고: 2021 이후 연도가 포함됨 — 평가 연도는 튜닝에 쓰지 않기로 했다 (decisions.md 원칙 1).")

    feat = pd.read_csv(ROOT / "out" / f"{args.code}_features.csv",
                       parse_dates=["date"], index_col="date")
    feat = add_extra_features(feat)         # plus10 컬럼 추가 (다른 세트에는 영향 없음)

    def check_set(name):
        missing = [c for c in FSETS[name] if c not in feat.columns]
        if missing:
            raise SystemExit(f"피처 세트 {name}: 컬럼 없음 {missing} — study/01_features/extra_features.py 를 먼저 실행")
        nan = feat[FSETS[name]].isna().sum()
        if nan.any():
            raise SystemExit(f"피처 세트 {name}: NaN 있음\n{nan[nan > 0]}")

    if args.sweep is None:
        check_set(args.features)
        df = run_config(feat, args, args.years, args.seeds, args.code)
        summarize(df, f"{args.features} w{args.window} h{args.hidden} "
                      f"lr{args.lr} γ{args.gamma} ep{args.episodes}")
    else:
        param, *values = args.sweep
        caster = {"episodes": int, "window": int, "hidden": int,
                  "lr": float, "gamma": float, "target_every": int,
                  "eps_decay": float, "features": str}[param]
        if param == "features":
            for v in values:
                check_set(v)
        results = []
        for v in values:
            setattr(args, param, caster(v))
            print(f"\n===== {param} = {v} =====")
            df = run_config(feat, args, args.years, args.seeds, args.code)
            results.append((v, *summarize(df, f"{param}={v}")))   # (값, 평균초과, 시드std, 폴드별, 거래)
        # ── 폴드별 순위 집계 (변동 큰 폴드가 평균을 지배하는 것을 보완) ──
        years_ = sorted(results[0][3])
        ranks = {r[0]: [] for r in results}
        for y in years_:
            order = sorted(results, key=lambda r: -r[3][y])          # 초과 높은 순
            for rank, r in enumerate(order, 1):
                ranks[r[0]].append(rank)
        print(f"\n{'='*60}\n스윕 결과 ({param}) — 규칙: ①평균 초과(동률 ±3%p) ②폴드 순위 ③시드 std ④기본값 유지")
        print(f"  {'값':>8} | {'평균 초과':>10} | {'시드 std':>8} | 폴드별 초과 " + " ".join(f"{y}" for y in years_) + " | 순위평균 | 거래/년")
        for v, m, sd, pf, tr in results:
            pfs = " ".join(f"{pf[y]:+.1%}" for y in years_)
            print(f"  {str(v):>8} | {m:+10.2%} | {sd:8.2%} | {pfs} | {sum(ranks[v])/len(ranks[v]):8.2f} | {tr:5.0f}")
        print("  (시드 std = 연도 내 시드 표준편차의 평균. 폴드 간 수익률 차이는 포함하지 않음)")
        print("→ 기본값 대비 평균 초과 +3%p 이상 & 3폴드 중 2폴드 이상 개선 & 2018(하락) 초과 비악화 → 변경. 아니면 기본값 유지.")
        print("→ 결과를 decisions.md 에 기록할 것.")
    print(f"\n전체 기록: {CSV}")


if __name__ == "__main__":
    main()
