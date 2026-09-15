"""00단계 (추가): 삼성전자 외부 데이터 — KOSPI 지수, (선택) VKOSPI, (선택) 투자자별 순매수.

실행:  python study/00_data/get_extra.py [--code 005930] [--start 2013-01-01]
결과:  out/kospi.csv              KOSPI 종가 (FinanceDataReader 'KS11')
       out/sp500.csv              S&P500 종가 (FDR 'US500' → 실패 시 yfinance '^GSPC') — 미국 날짜 기준, 정렬은 extra_features 에서
       out/vkospi.csv             VKOSPI 종가 — pykrx + KRX 로그인 필요, 실패하면 건너뜀
       out/<code>_flow.csv        투자자별 순매수 금액(원) — pykrx + KRX 로그인 필요, 실패하면 건너뜀

KRX 로그인: pykrx 최신 버전은 KRX 정보데이터시스템(data.krx.co.kr) 계정을 요구한다.
  회원가입 후 환경 변수  KRX_ID, KRX_PW  를 설정하고 실행한다 (PowerShell: $env:KRX_ID="..."; $env:KRX_PW="...").
  설정이 없으면 VKOSPI·수급은 건너뛰고 KOSPI 만 저장한다 — 피처 세트 market 은 그래도 동작하고, market_vk·flow 만 못 쓴다.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "out"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--code", default="005930")
ap.add_argument("--start", default="2013-01-01")
args = ap.parse_args()
start_compact = args.start.replace("-", "")
end_compact = pd.Timestamp.today().strftime("%Y%m%d")

# ── 1. KOSPI (FinanceDataReader) ─────────────────────────────────
import FinanceDataReader as fdr  # noqa: E402
ks = fdr.DataReader("KS11", args.start).rename(columns=str.lower)
ks.index.name = "date"
ks[["close"]].rename(columns={"close": "kospi"}).to_csv(OUT / "kospi.csv")
print(f"KOSPI {len(ks)}일 저장 → {OUT / 'kospi.csv'}  ({ks.index[0].date()} ~ {ks.index[-1].date()})")

# ── 1-b. S&P500 (KAIS 2021 논문의 해외 지수 피처용) ───────────────
sp = None
for sym in ["US500", "^GSPC", "SPY"]:
    try:
        d = fdr.DataReader(sym, args.start).rename(columns=str.lower)
        if len(d) > 1000:
            sp = d[["close"]].rename(columns={"close": "sp500"}); print(f"S&P500: FDR '{sym}' {len(sp)}일"); break
    except Exception as e:  # noqa: BLE001
        print(f"  FDR {sym} 실패: {str(e)[:80]}")
if sp is None:
    try:
        import yfinance as yf
        d = yf.download("^GSPC", start=args.start, auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        sp = d[["Close"]].rename(columns={"Close": "sp500"}); print(f"S&P500: yfinance ^GSPC {len(sp)}일")
    except Exception as e:  # noqa: BLE001
        print(f"  yfinance ^GSPC 실패: {e}")
if sp is not None:
    sp.index.name = "date"
    sp.to_csv(OUT / "sp500.csv")
    print(f"S&P500 저장 → {OUT / 'sp500.csv'}  ({sp.index[0].date()} ~ {sp.index[-1].date()})  ※ 미국 거래일 기준")
else:
    print("S&P500 미확보 → slim5_us 세트 사용 불가")

# ── 2. VKOSPI / 투자자별 순매수 (pykrx, 선택) ────────────────────
if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
    print("\nKRX_ID/KRX_PW 환경 변수가 없어 VKOSPI·투자자별 순매수는 건너뜀 (docstring 참고).")
    raise SystemExit(0)

try:
    from pykrx import stock
except ImportError:
    print("\npykrx 미설치: pip install pykrx  — VKOSPI·투자자별 순매수 건너뜀.")
    raise SystemExit(0)

# 2-a. VKOSPI: 지수 목록에서 이름으로 찾는다 (시장 구분에 따라 코드가 달라 이름 검색이 안전)
vk_ticker = None
for market in ["KRX", "KOSPI", "KOSDAQ", "테마"]:
    try:
        for t in stock.get_index_ticker_list(end_compact, market=market):
            name = stock.get_index_ticker_name(t)
            if "VKOSPI" in name.upper() or "변동성" in name:
                vk_ticker = t; print(f"VKOSPI 지수 발견: {t} {name} ({market})"); break
    except Exception as e:  # noqa: BLE001
        print(f"  지수 목록 조회 실패({market}): {e}")
    if vk_ticker:
        break
if vk_ticker:
    try:
        vk = stock.get_index_ohlcv_by_date(start_compact, end_compact, vk_ticker)
        vk.index.name = "date"
        vk[["종가"]].rename(columns={"종가": "vkospi"}).to_csv(OUT / "vkospi.csv")
        print(f"VKOSPI {len(vk)}일 저장 → {OUT / 'vkospi.csv'}")
    except Exception as e:  # noqa: BLE001
        print(f"VKOSPI 조회 실패: {e} — market_vk 세트는 사용 불가")
else:
    print("VKOSPI 지수를 목록에서 찾지 못함 — market_vk 세트는 사용 불가")

# 2-b. 투자자별 순매수 금액 (연 단위로 나눠 조회)
frames = []
for y in range(int(args.start[:4]), int(end_compact[:4]) + 1):
    try:
        f = stock.get_market_trading_value_by_date(f"{y}0101", f"{y}1231", args.code)
        frames.append(f)
        print(f"  수급 {y}: {len(f)}일")
    except Exception as e:  # noqa: BLE001
        print(f"  수급 {y} 조회 실패: {e}")
if frames:
    flow = pd.concat(frames).sort_index()
    flow.index.name = "date"
    cols = {c: c for c in flow.columns}
    ren = {}
    for c in flow.columns:
        if "외국인" in c: ren[c] = "frgn_net"
        elif "기관" in c and "합계" in c: ren[c] = "inst_net"
        elif c == "개인": ren[c] = "indiv_net"
    flow = flow.rename(columns=ren)
    keep = [c for c in ["frgn_net", "inst_net", "indiv_net"] if c in flow.columns]
    flow[keep].to_csv(OUT / f"{args.code}_flow.csv")
    print(f"투자자별 순매수 {len(flow)}일 저장 → {OUT / (args.code + '_flow.csv')}  컬럼 {keep}")
    print("  ※ 2015년 이전 행이 비어 있으면 KRX 제공 범위 밖 — flow 세트의 2018 폴드 학습이 짧아지므로 decisions.md 에 적을 것")
