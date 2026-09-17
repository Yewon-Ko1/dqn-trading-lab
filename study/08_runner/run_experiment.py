"""08단계: 실험 실행기. 빈칸 없음 — lesson 의 Double DQN 패치(dqn.py) 후 실행.

  python study/08_runner/run_experiment.py --quick     # 배관 점검
  python study/08_runner/run_experiment.py             # 본실험 (밤에 걸어두기)
  python study/08_runner/run_experiment.py --window 60 --features plus10 --hidden 64   # 09 튜닝 확정값으로

평가 구간 정합 (09-15 수정): 에이전트·MA교차 모두 테스트 연도 직전 거래일을 워밍업으로 받아
1월 첫 거래일부터 평가한다. 이전에는 에이전트가 window 일, MA교차가 60일을 건너뛴 채 시작해
연초 등락이 큰 해(2019·2020·2023)에서 B&H 대비 초과수익률이 최대 ±20%p 왜곡됐다.

산출물: out/experiments.csv (전 실행 기록, 이어쓰기)
        out/summary.md      (모델·기준선 × 지표, 평균±표준편차 — 논문 표 초안)
        out/fig_<code>_<model>_<year>.png (실행한 시드별 자산 곡선)
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
from split import make_folds  # noqa: E402
from metrics import cumulative_return, max_drawdown, sharpe, sortino  # noqa: E402
from baselines import buy_and_hold_pv, ma_crossover_pv, cash_pv, fixed_exposure_pv  # noqa: E402
from trading_env import TradingEnv  # noqa: E402
from dqn import DQNAgent  # noqa: E402
from tune import FSETS, CHURN, add_extra_features, with_warmup, data_fingerprint, same  # noqa: E402

CSV = ROOT / "out" / "experiments.csv"
COLS = ["time", "code", "model", "test_year", "seed", "episodes",
        "ret", "excess_bh", "sharpe", "sortino", "mdd", "trades",
        "window", "features", "hidden", "lr", "gamma", "train_len", "churn", "penalty",
        "n_step", "data_hash"]
# resume 의 설정 일치 판정에 쓰는 열 (data_hash 는 별도 비교)
CFG_NUM = ["episodes", "window", "hidden", "lr", "gamma", "train_len", "penalty", "n_step"]
CFG_STR = ["features", "churn"]


def metrics_row(model, year, seed, episodes, pv, trades, bh_ret, code="005930", cfg=None, data_hash=None):
    row = {"time": datetime.now().strftime("%m-%d %H:%M"), "code": code, "model": model,
           "test_year": year, "seed": seed, "episodes": episodes,
           "ret": round(cumulative_return(pv), 4),
           "excess_bh": round(cumulative_return(pv) - bh_ret, 4),
           "sharpe": round(sharpe(pv), 3), "sortino": round(sortino(pv), 3),
           "mdd": round(max_drawdown(pv), 4), "trades": trades}
    if cfg is not None:
        row.update({"window": cfg.window, "features": cfg.features, "hidden": cfg.hidden,
                    "lr": cfg.lr, "gamma": cfg.gamma, "train_len": cfg.train_len,
                    "churn": cfg.churn, "penalty": cfg.penalty, "n_step": cfg.n_step})
    row["data_hash"] = data_hash
    return row


def ma_crossover_in_year(close_ext: pd.Series, test_index, short=20, long=60):
    """MA교차 기준선을 워밍업 포함 시계열로 계산한 뒤 테스트 연도만 잘라 초기자본으로 재정규화."""
    pv_full, _ = ma_crossover_pv(close_ext, short=short, long=long)
    pv = pv_full.loc[test_index]
    pv = pv / pv.iloc[0] * 10_000_000
    signal = (close_ext.rolling(short).mean() > close_ext.rolling(long).mean())
    position = signal.shift(1, fill_value=False).loc[test_index].astype(int)
    trades = int((position.diff().fillna(0) != 0).sum())
    return pv, trades


def load_log() -> pd.DataFrame:
    """experiments.csv 를 COLS 형식으로 읽는다. 열이 추가되기 전의 옛 행은 기본값으로 채운다."""
    if not CSV.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(CSV)
    changed = list(df.columns) != COLS
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c, default in [("train_len", 3), ("churn", "none"), ("penalty", 0.0), ("n_step", 1)]:
        if df[c].isna().any():
            df[c] = df[c].fillna(default); changed = True
    df = df[COLS]
    if changed:
        df.to_csv(CSV, index=False)          # 형식 통일 (내용은 그대로)
    return df


def done_mask(log, code, model, year, cfg, data_hash):
    """이미 기록된 실행인지. cfg=None 이면 기준선(설정 무관, 데이터만 같으면 재사용)."""
    if len(log) == 0:
        return pd.Series([], dtype=bool)
    m = ((log["code"].astype(str).str.zfill(6) == str(code).zfill(6))   # CSV 는 앞 0 을 잃는다
         & (log["model"].astype(str) == str(model))
         & (pd.to_numeric(log["test_year"], errors="coerce") == year)
         & (log["data_hash"].astype(str) == str(data_hash)))
    if cfg is not None:
        for k in CFG_NUM:
            m &= same(log[k], getattr(cfg, k))
        for k in CFG_STR:
            m &= log[k].astype(str) == str(getattr(cfg, k))
    return m


def append_rows(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(CSV, mode="a", header=not CSV.exists(), index=False)


def run_agent(train_df, test_df, seed, episodes, double, cfg, feat):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fcols = FSETS[cfg.features]
    extra_state, min_hold = CHURN[cfg.churn]
    env = TradingEnv(train_df, window=cfg.window, features=fcols, extra_state=extra_state,
                     min_hold=min_hold, trade_penalty=cfg.penalty)
    agent = DQNAgent(state_dim=len(env.reset()), double=double, gamma=cfg.gamma, lr=cfg.lr,
                     hidden=cfg.hidden, target_every=cfg.target_every, eps_decay=cfg.eps_decay,
                     n_step=cfg.n_step)
    for _ in range(episodes):                       # 학습
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2
    test_ext = with_warmup(feat, test_df, cfg.window)   # 워밍업 window 일 + 테스트 연도 전체
    test_env = TradingEnv(test_ext, window=cfg.window, features=fcols, extra_state=extra_state,
                          min_hold=min_hold)                    # 테스트: ε 무시, 학습 없음 (패널티는 학습 신호라 불필요)
    s, done = test_env.reset(), False
    pv = [test_env.asset]
    while not done:
        s, _, done = test_env.step(agent.act(s, explore=False))
        pv.append(test_env.asset)
    pv = pd.Series(pv, index=test_ext.index[test_env.window:test_env.t + 1])
    assert pv.index[0] == test_df.index[0] and len(pv) == len(test_df), "평가 구간이 테스트 연도와 어긋남"
    return pv, test_env.trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dqn", "double"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--code", default="005930")
    ap.add_argument("--quick", action="store_true", help="fold 1 × seed 1 × ep 2 로 배관 점검")
    # 09 튜닝에서 확정한 값을 그대로 넘긴다 (기본값 = 튜닝 전 기준 설정)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--features", choices=list(FSETS), default="base8")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-every", type=int, default=500)
    ap.add_argument("--eps-decay", type=float, default=0.999)
    ap.add_argument("--train-len", type=int, default=3, help="학습 구간 연수, 0 = 가용 전체")
    ap.add_argument("--churn", choices=list(CHURN), default="none")
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--n-step", type=int, default=1,
                    help="n-step return (D-09). 1 이면 기존 1-step TD 와 동일")
    ap.add_argument("--resume", action="store_true",
                    help="같은 설정·같은 데이터로 이미 기록된 (연도, 모델, 시드) 는 건너뛰고 재사용")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.years, args.episodes = [1], [2024], 2

    feat = pd.read_csv(ROOT / "out" / f"{args.code}_features.csv",
                       parse_dates=["date"], index_col="date")
    feat = add_extra_features(feat)             # plus10 컬럼 (다른 세트에는 영향 없음)
    train_len = 30 if args.train_len == 0 else args.train_len
    folds = {te.index[0].year: (tr, te) for tr, te in make_folds(feat, args.years, train_len=train_len)}
    data_hash = data_fingerprint(feat, FSETS[args.features])
    log = load_log() if args.resume else pd.DataFrame(columns=COLS)
    if args.resume:
        print(f"resume: 기존 기록 {len(log)}행 | data_hash {data_hash}")

    all_rows, reused = [], 0
    for year, (train_df, test_df) in folds.items():
        close = test_df["close"]
        bh = buy_and_hold_pv(close); bh_ret = cumulative_return(bh)
        ma_pv, ma_tr = ma_crossover_in_year(with_warmup(feat, test_df, 60)["close"], test_df.index)
        base = []
        for name, pv_, tr_ in [("buy_hold", bh, None), ("ma_cross", ma_pv, ma_tr),
                               ("cash", cash_pv(close), 0), ("fixed50", fixed_exposure_pv(close), None)]:
            if args.resume:
                hit = log[done_mask(log, args.code, name, year, None, data_hash)]
                if len(hit):
                    all_rows += hit.to_dict(orient="records"); reused += len(hit); continue
            base.append(metrics_row(name, year, 0, 0, pv_, tr_, bh_ret, args.code, data_hash=data_hash))
        if base:
            append_rows(base); all_rows += base
        print(f"[{year}] 기준선 4종 (B&H {bh_ret:+.2%})")

        for model in args.models:
            curves = {}
            for seed in args.seeds:
                if args.resume:
                    m = done_mask(log, args.code, model, year, args, data_hash)
                    hit = log[m & (pd.to_numeric(log["seed"], errors="coerce") == seed)]
                    if len(hit):
                        all_rows += hit.to_dict(orient="records"); reused += len(hit); continue
                pv, trades = run_agent(train_df, test_df, seed, args.episodes,
                                       double=(model == "double"), cfg=args, feat=feat)
                row = metrics_row(model, year, seed, args.episodes, pv, trades, bh_ret, args.code,
                                  cfg=args, data_hash=data_hash)
                append_rows([row]); all_rows.append(row)
                curves[seed] = pv
                print(f"[{year}] {model} seed {seed}: {row['ret']:+.2%} "
                      f"(초과 {row['excess_bh']:+.2%}, 거래 {row['trades']})")
            if not curves:                          # 전부 재사용 → 그림은 이미 있음
                continue
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
             f"folds {args.years} × seeds {args.seeds} × episodes {args.episodes} | "
             f"window {args.window} features {args.features} hidden {args.hidden} lr {args.lr} γ {args.gamma} "
             f"train_len {args.train_len} churn {args.churn} penalty {args.penalty} "
             f"n_step {args.n_step} | data_hash {data_hash}", "",
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
    if reused:
        print(f"(resume) 기록된 {reused}행 재사용")
    print(f"\n요약: {out_md}\n전체 기록: {CSV}")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
