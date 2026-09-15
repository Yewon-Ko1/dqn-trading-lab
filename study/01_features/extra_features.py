"""01단계 (추가): D-03 피처 세트용 확장 피처. 빈칸 없음.

입력:  out/<code>_features.csv (01 features.py 산출물, OHLCV + base8)
       out/kospi.csv, (선택) out/vkospi.csv, (선택) out/<code>_flow.csv   ← 00 get_extra.py
출력:  out/<code>_features.csv 에 컬럼을 덧붙여 덮어쓴다 (base8 값은 그대로).

실행:  python study/01_features/extra_features.py [--code 005930]
검사:  python study/01_features/check_extra.py     (룩어헤드·NaN·스케일)

모든 피처는 비율형(가격 단위 무관)이고 t 일 값은 t 일까지의 데이터만 쓴다.

  [structure — 가격 구조, 장기 국면]
    close_ma200_ratio : close / MA200 - 1        200일선 이격도 — 강세/약세 국면의 고전적 기준
    close_hi252_ratio : close / max(close,252) - 1   52주 고점 대비 위치 (항상 ≤ 0) — "고점에서 얼마나 내려왔나"
    vol5_vol20        : std5(ret1) / std20(ret1) - 1   단기 변동성 급변 — 하락 초기에 튀는 경향
    hl_range          : (high - low) / close      일중 변동폭 — 종가만 쓰는 base8 이 버리는 정보
  [market — 시장 국면]
    kospi_ret5, kospi_ret20 : KOSPI 5·20일 수익률
    rel20                   : ret20 - kospi_ret20   시장 대비 상대강도
  [market_vk] (vkospi.csv 있을 때만)
    vkospi_ma20_ratio : VKOSPI / MA20(VKOSPI) - 1   공포지수의 평균 대비 수준
  [flow] (<code>_flow.csv 있을 때만)
    frgn5, frgn20 : 외국인 순매수금액 5·20일 합 / 거래대금(close×volume) 같은 기간 합
    inst20        : 기관 순매수금액 20일 합 / 거래대금 20일 합
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "out"

STRUCTURE = ["close_ma200_ratio", "close_hi252_ratio", "vol5_vol20", "hl_range"]
MARKET = ["kospi_ret5", "kospi_ret20", "rel20"]
MARKET_VK = ["vkospi_ma20_ratio"]
FLOW = ["frgn5", "frgn20", "inst20"]


def add_structure(f: pd.DataFrame) -> pd.DataFrame:
    close = f["close"]
    f["close_ma200_ratio"] = close / close.rolling(200).mean() - 1
    f["close_hi252_ratio"] = close / close.rolling(252).max() - 1
    r1 = close.pct_change()
    f["vol5_vol20"] = r1.rolling(5).std() / r1.rolling(20).std() - 1
    f["hl_range"] = (f["high"] - f["low"]) / close
    return f


def add_market(f: pd.DataFrame, kospi: pd.Series) -> pd.DataFrame:
    k = kospi.reindex(f.index)                      # 같은 KRX 거래일 — 빠지면 NaN 으로 남겨 검사에서 드러나게
    f["kospi_ret5"] = k.pct_change(5)
    f["kospi_ret20"] = k.pct_change(20)
    f["rel20"] = f["close"].pct_change(20) - f["kospi_ret20"]
    return f


def add_market_vk(f: pd.DataFrame, vk: pd.Series) -> pd.DataFrame:
    v = vk.reindex(f.index)
    f["vkospi_ma20_ratio"] = v / v.rolling(20).mean() - 1
    return f


def add_flow(f: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    fl = flow.reindex(f.index)
    value = (f["close"] * f["volume"]).astype(float)   # 거래대금 근사 (원)
    f["frgn5"] = fl["frgn_net"].rolling(5).sum() / value.rolling(5).sum()
    f["frgn20"] = fl["frgn_net"].rolling(20).sum() / value.rolling(20).sum()
    f["inst20"] = fl["inst_net"].rolling(20).sum() / value.rolling(20).sum()
    return f


def build(code: str) -> pd.DataFrame:
    f = pd.read_csv(OUT / f"{code}_features.csv", parse_dates=["date"], index_col="date").sort_index()
    f = f.drop(columns=[c for c in STRUCTURE + MARKET + MARKET_VK + FLOW if c in f.columns])  # 재실행 안전
    # structure 는 원본 OHLCV(<code>.csv) 로 계산해 features.py 가 버린 앞 59행도 워밍업에 쓴다
    raw = pd.read_csv(OUT / f"{code}.csv", parse_dates=["date"], index_col="date").sort_index()
    st = add_structure(raw.copy())[STRUCTURE]
    f = f.join(st, how="left")

    kospi = pd.read_csv(OUT / "kospi.csv", parse_dates=["date"], index_col="date")["kospi"]
    f = add_market(f, kospi)

    if (OUT / "vkospi.csv").exists():
        vk = pd.read_csv(OUT / "vkospi.csv", parse_dates=["date"], index_col="date")["vkospi"]
        f = add_market_vk(f, vk)
    else:
        print("vkospi.csv 없음 → market_vk 세트 미생성")

    if (OUT / f"{code}_flow.csv").exists():
        flow = pd.read_csv(OUT / f"{code}_flow.csv", parse_dates=["date"], index_col="date")
        f = add_flow(f, flow)
    else:
        print(f"{code}_flow.csv 없음 → flow 세트 미생성")

    # structure·market 은 필수: NaN 행 제거 (앞부분 워밍업 + 지수 결측일). 선택 세트는 NaN 을 남긴다.
    f = f.dropna(subset=STRUCTURE + MARKET)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
    code = ap.parse_args().code
    f = build(code)
    f.to_csv(OUT / f"{code}_features.csv")
    new = [c for c in STRUCTURE + MARKET + MARKET_VK + FLOW if c in f.columns]
    print(f"\n{len(f)}행 저장 ({f.index[0].date()} ~ {f.index[-1].date()}) → {OUT / (code + '_features.csv')}")
    print(f"추가 컬럼: {new}")
    print(f[new].describe().T[["mean", "std", "min", "max"]].round(4))
    nan = f[new].isna().sum()
    if nan.any():
        print("\nNaN 있는 선택 컬럼 (해당 세트는 그 구간에서 사용 불가):\n", nan[nan > 0])
