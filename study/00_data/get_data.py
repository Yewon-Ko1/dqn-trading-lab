"""00단계: 삼성전자(005930) 일봉 데이터를 CSV로 저장한다.

이 파일은 빈칸이 없다. 실행하고, 만들어진 CSV를 엑셀이나 메모장으로 열어 본다.
기존 backend/app/kis_client.py 는 같은 데이터를 한국투자증권 API에서 받아오는데,
토큰 발급·페이지 나누기 때문에 213줄이다. 우리는 공부용이므로 FinanceDataReader 로 5줄에 끝낸다.

실행:  python study/00_data/get_data.py            (시작일 기본 2013-01-01)
결과:  out/005930.csv
"""
import argparse
from pathlib import Path

import FinanceDataReader as fdr  # pip install finance-datareader

OUT = Path(__file__).resolve().parents[2] / "out"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
ap.add_argument("--start", default="2013-01-01",
                help="09-15부터 2013 시작: ma200·52주 고점 같은 장기 피처의 워밍업 확보 (fold 는 연도로 자르므로 학습 구간은 그대로)")
args = ap.parse_args(); code = args.code
# FinanceDataReader 의 국내 주식 소스(네이버)는 오늘 기준 최근 3,000행(약 12년)까지만 준다.
# 2013-01-01 을 요청해도 2014-06-26 부터 오므로, 그 앞은 다른 소스로 채운다:
#   1) KRX_ID/KRX_PW 가 있으면 pykrx (한국거래소 수정주가)   2) 아니면 yfinance ('005930.KS', 수정주가)
# 두 소스의 겹치는 구간 종가를 비교해 0.5% 이상 어긋나면 이어 붙이지 않고 멈춘다 (가격 단절 방지).
import os
import numpy as np
import pandas as pd

df = fdr.DataReader(code, args.start)
df = df.rename(columns=str.lower)
df.index.name = "date"
df = df[["open", "high", "low", "close", "volume"]]
print(f"FinanceDataReader: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행)")

# ── 데이터 정제 (09-15 추가) ──────────────────────────────────────
# (1) 액면분할: 네이버 소스는 가격만 소급 조정하고 거래량은 분할 전 주식 수 그대로 둔다.
#     삼성전자 2018-05-04 50:1 분할 → 그 전 거래량에 50 을 곱해 단위를 맞춘다 (안 하면 분할 후 20일간 volume_ma20_ratio 가 17배).
SPLITS = {"005930": [("2018-05-04", 50)],      # 삼성전자 50:1
          "035420": [("2018-10-12", 5)]}       # NAVER 5:1
# 그 외 종목: 아래 '분할 의심' 경고가 뜨면 확인 후 여기에 추가
for day, factor in SPLITS.get(code, []):
    before = df.index < pd.Timestamp(day)
    df.loc[before, "volume"] = df.loc[before, "volume"] * factor
    print(f"  액면분할 조정: {day} 이전 거래량 ×{factor} ({before.sum()}행)")
# (1-b) 분할 의심: 거래량이 직전 20일 평균의 15배를 넘는 날 — 조정 안 된 분할이면 여기 걸린다 (이벤트일 수도 있으니 눈으로 확인)
sus = df.index[(df["volume"] > 15 * df["volume"].rolling(20).mean().shift(1)).fillna(False)]
if len(sus):
    print(f"  ⚠️ 분할 의심(거래량 급증) 날짜: {[d.date().isoformat() for d in sus]} — 분할이면 SPLITS 에 추가")
# (2) 매매정지일(시가·거래량 0 — 분할 전 3일 등)은 거래가 불가능한 날이므로 행을 뺀다.
halt = (df["open"] == 0) | (df["volume"] == 0)
if halt.any():
    print(f"  매매정지 행 제거: {list(df.index[halt].date)}")
    df = df[~halt]


def backfill(start: str, end: pd.Timestamp) -> pd.DataFrame | None:
    """start ~ end 구간 OHLCV 를 pykrx 또는 yfinance 로 받는다. 실패하면 None."""
    if os.environ.get("KRX_ID") and os.environ.get("KRX_PW"):
        try:
            from pykrx import stock
            k = stock.get_market_ohlcv(start.replace("-", ""), end.strftime("%Y%m%d"), code, adjusted=True)
            k = k.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
            k.index = pd.to_datetime(k.index); k.index.name = "date"
            print(f"  보충 소스: pykrx {k.index[0].date()} ~ {k.index[-1].date()} ({len(k)}행)")
            return k[["open", "high", "low", "close", "volume"]]
        except Exception as e:  # noqa: BLE001
            print(f"  pykrx 실패: {e}")
    try:
        import yfinance as yf  # pip install yfinance
        # auto_adjust=False: 야후의 Open/High/Low/Close 는 액면분할만 조정된 값 (Adj Close 는 배당까지 조정 → FDR 과 기준이 다름)
        y = yf.download(f"{code}.KS", start=start, end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        auto_adjust=False, progress=False)
        if isinstance(y.columns, pd.MultiIndex):
            y.columns = y.columns.get_level_values(0)
        y = y.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        y.index = pd.to_datetime(y.index).tz_localize(None); y.index.name = "date"
        y = y[y["volume"] > 0]                       # 야후는 휴장일에 거래량 0 행을 넣을 때가 있다
        print(f"  보충 소스: yfinance {y.index[0].date()} ~ {y.index[-1].date()} ({len(y)}행)")
        return y
    except Exception as e:  # noqa: BLE001
        print(f"  yfinance 실패: {e} (pip install yfinance)")
    return None


gap_days = (df.index[0] - pd.Timestamp(args.start)).days
if gap_days > 30:
    print(f"⚠️  첫 행이 {df.index[0].date()} — 요청 시작일 {args.start} 보다 {gap_days}일 늦음 → 앞부분 보충 시도")
    overlap_end = df.index[min(60, len(df) - 1)]           # 겹침 검증용으로 60거래일 더 받는다
    old = backfill(args.start, overlap_end)
    if old is not None:
        common = old.index.intersection(df.index)
        if len(common) < 20:
            raise SystemExit("보충 소스와 겹치는 날이 20일 미만 — 이어 붙이지 않음")
        ratio = (old.loc[common, "close"] / df.loc[common, "close"])
        med, spread = float(ratio.median()), float(ratio.max() - ratio.min())
        vmed = float((old.loc[common, "volume"] / df.loc[common, "volume"]).median())
        print(f"  겹침 {len(common)}일 종가 비율: 중앙값 {med:.4f}, 범위 {spread:.4f} | 거래량 비율 중앙값 {vmed:.3f}")
        if spread > 0.01:
            raise SystemExit("보충 소스와 FDR 의 종가 비율이 날마다 다름(범위 > 1%) — 조정 기준이 달라 이어 붙이지 않음. "
                             "KRX 로그인(pykrx) 으로 다시 시도하거나 --start 2014-06-26 으로 실행")
        if abs(med - 1) > 0.005:
            print(f"  ⚠️ 비율이 상수 {med:.4f} — 배당 조정 여부 차이로 보임. 비율로 맞춰 이어 붙인다 "
                  f"(보충 구간의 배당락일 근처에 1% 미만 오차 가능, 워밍업 용도라 허용)")
        old = old.loc[old.index < df.index[0]]
        for c in ["open", "high", "low", "close"]:
            old[c] = old[c] / med                         # 미세한 기준 차이는 비율로 맞춘다
        if abs(vmed - 1) > 0.05:
            print(f"  ⚠️ 거래량 단위가 다름(비율 {vmed:.2f}) → 보충 구간 거래량을 비율로 맞춘다")
            old["volume"] = old["volume"] / vmed
        df = pd.concat([old, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(0).astype(int)
        df["volume"] = df["volume"].astype(int)
        print(f"  보충 후: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행)")

df.to_csv(OUT / f"{code}.csv")
if (df.index[0] - pd.Timestamp(args.start)).days > 30:
    print(f"⚠️  여전히 첫 행이 {df.index[0].date()} — 장기 피처(ma200·52주 고점) 워밍업이 부족해 2015년 일부가 잘린다")
print(df.head())
print(df.tail())
print(f"\n{len(df)}일치 저장 → {OUT / (code + '.csv')}")

# ── 스스로 답해 볼 것 (notes.md 에 적기) ─────────────────────────────
# Q1. 행 하나는 무엇을 뜻하나? 주말/공휴일 행이 있나?
# Q2. close 가 2015년엔 2만원대인데 2018년에 갑자기 5만원대로 바뀌는 구간이 있나? (액면분할)
#     → 이런 '가격 자체'를 모델에 넣으면 왜 문제가 되는지 01단계에서 다룬다.
# Q3. volume 의 단위는? 값의 크기가 close 와 몇 배 차이 나나?
