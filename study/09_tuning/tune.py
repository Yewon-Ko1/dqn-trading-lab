"""09단계: 하이퍼파라미터 튜닝 실행기. 빈칸 없음.

원칙 (decisions.md 참고):
  1. 튜닝은 검증 폴드(기본: 테스트 2018 하락·2019 상승·2020 V자)에서만 한다. 평가 연도 2021~2025와 2026 보조 구간은
     설정 확정 후 딱 한 번만 다시 돌린다 — 테스트 성적을 보고 고르는 순간 체리피킹이다.
  2. 한 번에 한 변수만 바꾼다. 모든 실행은 설정 전체와 함께 out/tuning.csv 에 기록된다.
  3. 자동 체인은 시드 20개 평균 ± 표준편차와 폴드별 결과를 보고한다. 동률 기준 ±3%p, 기본값 유지가 디폴트.

사용 예:
  python study/09_tuning/tune.py                              # 기준 설정, 검증 폴드
  python study/09_tuning/tune.py --episodes 20
  python study/09_tuning/tune.py --sweep episodes 10 20 40    # 한 변수 스윕
  python study/09_tuning/tune.py --sweep window 10 20 60
  python study/09_tuning/tune.py --window 10 --sweep features slim5 base8 plus10 structure market flow
  python study/09_tuning/tune.py --resume --window 10 --sweep features slim3 slim7 slim5_vol slim5_us core8   # D-03b
  python study/09_tuning/tune.py --sweep hidden 32 64 128
  python study/09_tuning/tune.py --sweep lr 3e-4 1e-3 3e-3
  python study/09_tuning/tune.py --sweep gamma 0.95 0.99 0.995
  python study/09_tuning/tune.py --resume --sweep features ...   # 중단된 스윕 이어 돌리기 (완료된 실행은 건너뜀)
  python study/09_tuning/tune.py --resume --window 10 --features slim5 --sweep train_len 3 5 0     # D-03.7 학습 구간
  python study/09_tuning/tune.py --resume --window 10 --features slim5 --sweep churn none state hold3 hold10   # D-03.5

산출물: out/tuning.csv (전 실행 기록), 실행 끝에 설정별 요약표 출력.
"""
import argparse
import hashlib
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
        "ret", "excess_bh", "sortino", "mdd", "trades", "data_hash", "train_len", "churn", "penalty"]
CFG_KEYS = ["episodes", "window", "features", "hidden", "lr", "gamma", "target_every", "eps_decay",
            "train_len", "churn", "penalty"]

# D-03.5 잦은 매매 변형: (extra_state, min_hold)
CHURN = {"none": (False, 0), "state": (True, 0), "hold3": (False, 3), "hold10": (False, 10),
         "hold20": (False, 20), "both": (True, 3)}


def data_fingerprint(feat: pd.DataFrame, cols) -> str:
    """실제로 쓰는 열(피처 세트 + OHLCV)의 값 해시 앞 10자리. 데이터가 바뀌면 값이 바뀌어 옛 행을 재사용하지 않게 하고,
    쓰지 않는 열이 추가되는 것만으로는 바뀌지 않아 resume 재사용이 유지된다."""
    use = list(cols) + ["open", "high", "low", "close", "volume"]
    arr = np.ascontiguousarray(feat[use].to_numpy(dtype=np.float64))
    return hashlib.md5(arr.tobytes() + str(list(feat.index[[0, -1]])).encode()).hexdigest()[:10]


def same(col: pd.Series, value) -> pd.Series:
    """설정값 비교: 숫자는 숫자로(3 == 3.0, 1e-3 == 0.001), 문자열은 문자열로."""
    try:
        v = float(value)
        return pd.to_numeric(col, errors="coerce") == v
    except (TypeError, ValueError):
        return col.astype(str) == str(value)


def load_log() -> pd.DataFrame:
    """tuning.csv 를 읽는다. 열이 추가된 뒤의 옛 파일(data_hash 없음)은 빈 값으로 채워 형식을 맞춘다."""
    if not CSV.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(CSV)
    changed = list(df.columns) != COLS
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    # 옵션이 생기기 전의 행은 정의상 기본값으로 실행된 것 → 기본값으로 채워야 resume 이 재사용한다
    for c, default in [("train_len", 3), ("churn", "none"), ("penalty", 0.0)]:
        if df[c].isna().any():
            df[c] = df[c].fillna(default); changed = True
    df = df[COLS]
    if changed:
        df.to_csv(CSV, index=False)          # 형식 통일 (내용은 그대로)
    return df

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
    # ── D-03b (2라운드, 09-15 사전 등록) ──
    "slim3": ["ret1", "ret5", "vol20"],                                          # 대조군: 최소 표현 (TDQN 식)
    "slim7": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio",
              "close_ma200_ratio", "close_hi252_ratio"],                         # slim5 + 장기 국면 (1라운드 structure 패턴)
    "slim5_vol": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio",
                  "vol_regime", "z_ret", "atr14_ratio"],                         # slim5 + 변동성 국면 (FinRL 터뷸런스·KAIS ATR)
    "slim5_us": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio",
                 "sp_ret1", "sp_ret5"],                                          # slim5 + 미국 지수 t-1 (KAIS 2021, sp500.csv 필요)
    "core8": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio",
              "close_ma200_ratio", "frgn20", "hl_range"],                        # 계열별 대표 1개 (flow.csv 필요)
    # ── D-03.9 국면 조건부 (09-15 사전 등록) ──
    "slim7_gated": ["ret1", "ret5", "close_ma20_ratio", "vol20", "volume_ma20_ratio",
                    "close_ma200_ratio_g", "close_hi252_ratio_g"],              # 변동성 스트레스 국면에서만 장기 피처
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
    return out.dropna(subset=["rsi14", "macd_ratio"])   # 다른 세트의 NaN(선택 열) 때문에 시작 행이 바뀌지 않도록


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
    extra_state, min_hold = CHURN[cfg.churn]
    env = TradingEnv(train_df, window=cfg.window, features=fcols, extra_state=extra_state, min_hold=min_hold,
                     trade_penalty=cfg.penalty)
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
    test_env = TradingEnv(test_ext, window=cfg.window, features=fcols, extra_state=extra_state, min_hold=min_hold)
    s, done = test_env.reset(), False
    pv = [test_env.asset]                                        # t=window = 테스트 연도 첫 거래일
    while not done:
        s, _, done = test_env.step(agent.act(s, explore=False))
        pv.append(test_env.asset)
    pv = pd.Series(pv, index=test_ext.index[test_env.window:test_env.t + 1])
    assert pv.index[0] == test_df.index[0] and len(pv) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    return pv, test_env.trades


def run_config(feat, cfg, years, seeds, code, data_hash=None, resume=False):
    """설정 하나를 years × seeds 로 실행. resume=True 면 같은 설정·데이터로 이미 기록된 (연도, 시드) 는 건너뛰고 그 행을 재사용."""
    data_hash = data_fingerprint(feat, FSETS[cfg.features])          # 피처 세트별 지문
    done = pd.DataFrame(columns=COLS)
    if resume:
        log = load_log()
        mask = (log["code"].astype(str).str.zfill(6) == str(code).zfill(6)) & (log["data_hash"] == data_hash)  # CSV 는 앞 0 을 잃는다
        for k in CFG_KEYS:
            mask &= same(log[k], getattr(cfg, k))
        done = log[mask].drop_duplicates(subset=["test_year", "seed"], keep="last")
        if len(done):
            print(f"  (resume) 기록된 {len(done)}개 실행 재사용")
    rows = []
    train_len = 30 if cfg.train_len == 0 else cfg.train_len      # 0 = 가용 전체 (2013~)
    folds = {te.index[0].year: (tr, te) for tr, te in make_folds(feat, years, train_len=train_len)}
    for year, (train_df, test_df) in folds.items():
        bh_ret = cumulative_return(buy_and_hold_pv(test_df["close"]))
        for seed in seeds:
            if ((done["test_year"] == year) & (done["seed"] == seed)).any():
                continue
            pv, trades = run_one(train_df, test_df, seed, cfg, feat)
            ret = cumulative_return(pv)
            row = {"time": datetime.now().strftime("%m-%d %H:%M"), "code": code,
                   "test_year": year, "seed": seed,
                   "episodes": cfg.episodes, "window": cfg.window,
                   "features": cfg.features, "hidden": cfg.hidden,
                   "lr": cfg.lr, "gamma": cfg.gamma,
                   "target_every": cfg.target_every, "eps_decay": cfg.eps_decay,
                   "ret": round(ret, 4), "excess_bh": round(ret - bh_ret, 4),
                   "sortino": round(sortino(pv), 3),
                   "mdd": round(max_drawdown(pv), 4), "trades": trades, "data_hash": data_hash,
                   "train_len": cfg.train_len, "churn": cfg.churn, "penalty": cfg.penalty}
            pd.DataFrame([row], columns=COLS).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)  # 한 줄씩 즉시 기록
            rows.append(row)
            print(f"  [{year}] seed {seed}: {ret:+.2%} (초과 {ret - bh_ret:+.2%}, 거래 {trades})")
    new = pd.DataFrame(rows, columns=COLS)
    out = pd.concat([done, new], ignore_index=True) if len(done) else new
    out = out[out["test_year"].isin(years) & out["seed"].isin(seeds)]
    return out.sort_values(["test_year", "seed"]).reset_index(drop=True)


def summarize(df, label):
    g = df.groupby("test_year")
    exc_m, exc_s = df["excess_bh"].mean(), df["excess_bh"].std()
    print(f"\n■ {label}")
    for y, gy in g:
        print(f"  {y}: 초과 {gy['excess_bh'].mean():+.2%} ± {gy['excess_bh'].std():.2%} "
              f"(승률 {(gy['excess_bh'] > 0).sum()}/{len(gy)})  MDD {gy['mdd'].mean():.2%}")
    print(f"  종합: 초과 {exc_m:+.2%} ± {exc_s:.2%}  |  시드 표준편차(연도 내 평균) "
          f"{g['ret'].std().mean():.2%}  |  거래 {df['trades'].mean():.0f}회  |  거래 0회(퇴화) {(df['trades'] == 0).sum()}/{len(df)}")
    per_fold = {int(y): gy["excess_bh"].mean() for y, gy in g}
    seed_sd = g["ret"].std().mean()          # 연도 내 시드 std 의 평균 (폴드 간 차이는 제외)
    return exc_m, seed_sd, per_fold, df["trades"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020],
                    help="튜닝 폴드(하락·상승·V자). 평가 연도(2021~2025·2026)는 확정 후에만!")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--features", choices=list(FSETS), default="base8")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-every", type=int, default=500)
    ap.add_argument("--eps-decay", type=float, default=0.999)
    ap.add_argument("--train-len", type=int, default=3,
                    help="학습 구간 연수 (D-03.7). 0 = 가용 전체(2013~). 기본 3")
    ap.add_argument("--penalty", type=float, default=0.0,
                    help="D-03.6 매매 1회당 보상 패널티 (학습 신호만, 자산 무관). 기본 0")
    ap.add_argument("--churn", choices=list(CHURN), default="none",
                    help="D-03.5 잦은 매매 변형: none | state(보유일수·손익 상태) | hold3 | hold10 | hold20 | both")
    ap.add_argument("--sweep", nargs="+", default=None,
                    metavar=("PARAM", "VALUES"),
                    help="예: --sweep episodes 10 20 40 (그 외 인자는 고정값으로 사용)")
    ap.add_argument("--resume", action="store_true",
                    help="같은 설정·같은 데이터로 이미 tuning.csv 에 기록된 (연도, 시드) 는 건너뛴다 — 중단된 스윕 이어 돌리기")
    args = ap.parse_args()

    if any(y >= 2021 for y in args.years):
        print("⚠️  경고: 2021 이후 연도가 포함됨 — 평가 연도는 튜닝에 쓰지 않기로 했다 (decisions.md 원칙 1).")

    feat = pd.read_csv(ROOT / "out" / f"{args.code}_features.csv",
                       parse_dates=["date"], index_col="date")
    feat = add_extra_features(feat)         # plus10 컬럼 추가 (다른 세트에는 영향 없음)
    load_log()                              # 옛 형식 파일이면 열을 맞춰 둔다
    print(f"데이터 지문(기본 세트 {args.features}) {data_fingerprint(feat, FSETS[args.features])}")

    def check_set(name):
        missing = [c for c in FSETS[name] if c not in feat.columns]
        if missing:
            raise SystemExit(f"피처 세트 {name}: 컬럼 없음 {missing} — study/01_features/extra_features.py 를 먼저 실행")
        nan = feat[FSETS[name]].isna().sum()
        if nan.any():
            raise SystemExit(f"피처 세트 {name}: NaN 있음\n{nan[nan > 0]}")

    if args.sweep is None:
        check_set(args.features)
        df = run_config(feat, args, args.years, args.seeds, args.code, resume=args.resume)
        summarize(df, f"{args.features} w{args.window} h{args.hidden} "
                      f"lr{args.lr} γ{args.gamma} ep{args.episodes}")
    else:
        param, *values = args.sweep
        caster = {"episodes": int, "window": int, "hidden": int,
                  "lr": float, "gamma": float, "target_every": int,
                  "eps_decay": float, "features": str,
                  "train_len": int, "churn": str, "penalty": float}[param]
        if param == "features":
            for v in values:
                check_set(v)
        results = []
        for v in values:
            setattr(args, param, caster(v))
            print(f"\n===== {param} = {v} =====")
            df = run_config(feat, args, args.years, args.seeds, args.code, resume=args.resume)
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
