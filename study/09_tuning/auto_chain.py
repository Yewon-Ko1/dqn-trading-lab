"""09단계: 무인 튜닝 체인 (72시간 예산용). 빈칸 없음.

decisions.md 의 결정 규칙을 그대로 코드로 옮겨, 스윕 → 규칙 판정 → 다음 스윕을 사람 개입 없이 잇는다.
사람이 결과를 보고 개입할 여지를 없애는 것이 목적이므로, 규칙은 decisions.md 와 동일해야 한다.

  python study/09_tuning/auto_chain.py                 # 전체 체인 (중단되면 같은 명령으로 이어 감 — 모든 실행은 --resume)
  python study/09_tuning/auto_chain.py --quick         # 배관 점검 (시드 1, 에피소드 1, 폴드 2019)
  python study/09_tuning/auto_chain.py --stages D-03.7 D-04   # 일부 단계만

단계 (사전 등록 09-15, 시드 1~20):
  D-03.7 train_len 3/5/0       기준 3    동률 시 짧은 쪽
  D-03.6 (churn, penalty) 6조합 기준 (hold10, 0)  동률 시 penalty 작은 쪽 → churn none(학습) 우선
  D-03.9 features slim5/slim7/slim7_gated  기준 slim5  동률 시 단순한 쪽 (국면 조건부 피처)
  D-04   hidden 32/64/128/256  기준 64   동률 시 작은 쪽
  D-05   lr 3e-4/1e-3/3e-3     기준 1e-3 동률 시 기준
  D-06   gamma 0.95/0.99/0.995 기준 0.99 동률 시 기준
  D-07   episodes 10/20/40     기준 10   동률 시 작은 쪽  (제약 도입으로 학습 역학이 바뀌어 재확인)
  D-99   run_experiment.py 2021~2025 + 2026 보조 — 삼성전자 시드 1~20, dqn·double
  D-99a  절제: 삼성전자 확정 설정에서 churn none (제약 효과가 평가 연도에서도 재현되는지)
  (보조) KODEX200 069500 — features CSV 가 있으면 시드 1~10 으로 같이 실행 (시장 지수 보조 실험)
  D-99b  다종목 일반화는 이번 논문에서 제외 (10월). --generalize 를 붙일 때만 2020년 말 시총 상위 5개 실행

규칙 5 (기본값 대비): (a) 평균 초과 +3%p 이상  (b) 3폴드 중 2폴드 이상 개선  (c) 2018 초과가 3%p 넘게 악화되지 않음.
통과 후보 중 평균 초과 1위. 1위와 3%p 이내인 통과 후보가 있으면 '단순한 쪽'(단계별 prefer 순서). 통과 없으면 기본값 유지.

산출물: out/auto_chain.md (단계별 표·판정·확정 설정), out/tuning.csv (원자료), out/experiments.csv (본실험)
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "study" / "09_tuning"))
TUNE = ROOT / "study" / "09_tuning" / "tune.py"
RUNX = ROOT / "study" / "08_runner" / "run_experiment.py"
CSV = ROOT / "out" / "tuning.csv"
LOG = ROOT / "out" / "auto_chain.md"
YEARS = [2018, 2019, 2020]
SEEDS = list(range(1, 21))
TIE = 0.03

# 체인 시작 설정 (D-01~D-03.5 확정값)
BASE = {"episodes": 10, "window": 10, "features": "slim5", "hidden": 64, "lr": 1e-3, "gamma": 0.99,
        "target_every": 500, "eps_decay": 0.999, "train_len": 3, "churn": "hold10", "penalty": 0.0}

# 단계 정의: name, 바꾸는 키(들), 후보(dict 목록), prefer(단순한 순서, 인덱스 작을수록 단순)
STAGES = [
    {"name": "D-03.7", "keys": ["train_len"],
     "cands": [{"train_len": 3}, {"train_len": 5}, {"train_len": 0}],
     "prefer": [{"train_len": 3}, {"train_len": 5}, {"train_len": 0}]},
    {"name": "D-03.6", "keys": ["churn", "penalty"],
     "cands": [{"churn": c, "penalty": p} for c in ["hold10", "none"] for p in [0.0, 0.003, 0.01]],
     # 동률이면: 패널티 작은 쪽, 그 다음 none(학습)보다 hold10(확정값) — 기본값 유지 원칙
     "prefer": [{"churn": "hold10", "penalty": 0.0}, {"churn": "none", "penalty": 0.003}, {"churn": "hold10", "penalty": 0.003},
                {"churn": "none", "penalty": 0.01}, {"churn": "hold10", "penalty": 0.01}, {"churn": "none", "penalty": 0.0}]},
    {"name": "D-03.9", "keys": ["features"],
     "cands": [{"features": f} for f in ["slim5", "slim7", "slim7_gated"]],
     "prefer": [{"features": f} for f in ["slim5", "slim7", "slim7_gated"]]},
    {"name": "D-04", "keys": ["hidden"],
     "cands": [{"hidden": h} for h in [32, 64, 128, 256]],
     "prefer": [{"hidden": h} for h in [32, 64, 128, 256]]},
    {"name": "D-05", "keys": ["lr"],
     "cands": [{"lr": v} for v in [3e-4, 1e-3, 3e-3]],
     "prefer": [{"lr": 1e-3}, {"lr": 3e-4}, {"lr": 3e-3}]},
    {"name": "D-06", "keys": ["gamma"],
     "cands": [{"gamma": v} for v in [0.95, 0.99, 0.995]],
     "prefer": [{"gamma": 0.99}, {"gamma": 0.95}, {"gamma": 0.995}]},
    {"name": "D-07", "keys": ["episodes"],
     "cands": [{"episodes": e} for e in [10, 20, 40]],
     "prefer": [{"episodes": e} for e in [10, 20, 40]]},
]


def log(text: str):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def cfg_args(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def run_tune(cfg: dict, seeds, years, quick=False):
    cmd = [sys.executable, str(TUNE), "--resume", *cfg_args(cfg),
           "--years", *map(str, years), "--seeds", *map(str, seeds)]
    if quick:
        cmd += ["--episodes", "1"]
    print("$", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)


def rows_for(cfg: dict, seeds, years) -> pd.DataFrame:
    from tune import load_log, same
    df = load_log()
    m = (df["code"].astype(str).str.zfill(6) == "005930")
    for k, v in cfg.items():
        m &= same(df[k], v)
    d = df[m & df["test_year"].isin(years) & df["seed"].isin(seeds)]
    return d.drop_duplicates(["test_year", "seed"], keep="last")


def summarize(d: pd.DataFrame) -> dict:
    g = d.groupby("test_year")
    return {"mean": d["excess_bh"].mean(), "years": {int(y): gy["excess_bh"].mean() for y, gy in g},
            "std": g["ret"].std().mean(), "trades": d["trades"].mean(), "mdd": d["mdd"].mean(),
            "n": len(d), "degen": int((d["trades"] == 0).sum())}


def passes(c: dict, ref: dict) -> tuple[bool, str]:
    a = c["mean"] - ref["mean"] >= TIE
    improved = sum(c["years"][y] > ref["years"][y] for y in ref["years"])
    b = improved >= 2
    cc = c["years"].get(2018, 0) >= ref["years"].get(2018, 0) - TIE
    return (a and b and cc), f"(a){c['mean'] - ref['mean']:+.1%}{'✓' if a else '✗'} (b){improved}/3{'✓' if b else '✗'} (c){'✓' if cc else '✗'}"


def decide(stage: dict, base: dict, results: list[tuple[dict, dict]]) -> dict:
    """results: [(override, summary)]. 기본값 = base 의 해당 키 값."""
    default = {k: base[k] for k in stage["keys"]}
    ref = next(s for o, s in results if all(str(o[k]) == str(default[k]) for k in stage["keys"]))
    passers = []
    lines = []
    for o, s in results:
        is_ref = all(str(o[k]) == str(default[k]) for k in stage["keys"])
        ok, why = (False, "기준") if is_ref else passes(s, ref)
        ys = " ".join(f"{s['years'][y]:+.1%}" for y in sorted(s["years"]))
        lines.append(f"| {o} | {s['mean']:+.1%} | {s['std']:.1%} | {ys} | {s['trades']:.0f} | {s['degen']} | {why} |")
        if ok:
            passers.append((o, s))
    log("| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (" + " ".join(map(str, sorted(ref["years"]))) + ") | 거래/년 | 퇴화 | 규칙 5 |")
    log("|---|---|---|---|---|---|---|")
    for ln in lines:
        log(ln)
    if not passers:
        log(f"→ 통과 후보 없음 → **기본값 유지** {default}")
        return default
    top = max(s["mean"] for _, s in passers)
    near = [o for o, s in passers if s["mean"] >= top - TIE]
    order = [str(p) for p in stage["prefer"]]
    near.sort(key=lambda o: order.index(str({k: o[k] for k in stage["keys"]})) if str({k: o[k] for k in stage["keys"]}) in order else 99)
    chosen = {k: near[0][k] for k in stage["keys"]}
    log(f"→ 통과 {len(passers)}개, 1위 {top:+.1%}, 3%p 이내 {len(near)}개 → 단순한 쪽 **{chosen}** 채택")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--final-seeds", nargs="+", type=int, default=list(range(1, 21)),
                    help="D-99 삼성전자 시드 (기본 1~20). D-99b 일반화 종목은 앞 10개만")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--generalize", action="store_true",
                    help="D-99b 다종목 일반화까지 실행 (이번 논문 제외, 10월용)")
    args = ap.parse_args()
    seeds, years = ([1], [2019]) if args.quick else (SEEDS, YEARS)

    cfg = dict(BASE)
    log(f"\n# 무인 체인 시작 {datetime.now():%m-%d %H:%M}  시드 {seeds[0]}~{seeds[-1]}  폴드 {years}\n시작 설정: {cfg}")
    for stage in STAGES:
        if args.stages and stage["name"] not in args.stages:
            continue
        log(f"\n## {stage['name']}  ({datetime.now():%m-%d %H:%M})  고정: { {k: v for k, v in cfg.items() if k not in stage['keys']} }")
        results = []
        for o in stage["cands"]:
            c = {**cfg, **o}
            run_tune(c, seeds, years, quick=args.quick)
            d = rows_for({**c, **({"episodes": 1} if args.quick else {})}, seeds, years)
            if len(d) == 0:
                raise SystemExit(f"{stage['name']} {o}: tuning.csv 에 결과 없음")
            results.append((o, summarize(d)))
        chosen = decide(stage, cfg, results)
        cfg.update(chosen)
        log(f"확정 설정: {cfg}")

    log(f"\n## 최종 설정 ({datetime.now():%m-%d %H:%M}): {cfg}")
    if args.skip_final or args.quick:
        return
    # ── D-99 본실험: 확정 설정 1회 ──
    years = ["2021", "2022", "2023", "2024", "2025", "2026"]

    def run_final(code, seeds, override=None, label=""):
        if not (ROOT / "out" / f"{code}_features.csv").exists():
            log(f"({label}) {code}_features.csv 없음 → 건너뜀"); return
        c = {**cfg, **(override or {})}
        cmd = [sys.executable, str(RUNX), "--code", code, "--models", "dqn", "double",
               "--years", *years, "--seeds", *map(str, seeds), *cfg_args(c)]
        log(f"$ ({label}) {' '.join(cmd[1:])}")
        subprocess.run(cmd, check=True)

    run_final("005930", args.final_seeds, label="D-99 본실험")
    run_final("005930", args.final_seeds, {"churn": "none", "penalty": 0.0}, label="D-99a 절제(제약 없음)")
    run_final("069500", args.final_seeds[:10], label="보조 KODEX200")
    if args.generalize:                                                             # 10월: SK하이닉스, LG화학, 삼성바이오, 셀트리온, NAVER
        for code in ["000660", "051910", "207940", "068270", "035420"]:
            run_final(code, args.final_seeds[:10], label="D-99b 일반화")
    log(f"\n# 체인 종료 {datetime.now():%m-%d %H:%M}. 결과: out/experiments.csv, out/summary.md")


if __name__ == "__main__":
    main()
