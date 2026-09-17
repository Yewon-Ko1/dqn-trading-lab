"""01단계 (추가): D-03 피처 세트용 확장 피처. 빈칸 없음.

입력:  out/<code>_features.csv (01 features.py 산출물, OHLCV + base8)
       out/<code>.csv (원본 OHLCV — 장기 피처 워밍업용)
       out/kospi.csv, (선택) out/sp500.csv, out/vkospi.csv, out/<code>_flow.csv   ← 00 get_extra.py
출력:  out/<code>_features.csv 에 컬럼을 덧붙여 덮어쓴다 (base8 값은 그대로).

실행:  python study/01_features/extra_features.py [--code 005930]
검사:  python study/01_features/check_extra.py     (룩어헤드·NaN·스케일·미국 지수 정렬)

모든 피처는 비율형(가격 단위 무관)이고 t 일 값은 t 일까지의 데이터만 쓴다.

  [structure — 가격 구조, 장기 국면]  (D-03 1라운드)
    close_ma200_ratio : close / MA200 - 1        200일선 이격도 — 강세/약세 국면의 고전적 기준
    close_hi252_ratio : close / max(close,252) - 1   52주 고점 대비 위치 (항상 ≤ 0)
    vol5_vol20        : std5(ret1) / std20(ret1) - 1   단기 변동성 급변
    hl_range          : (high - low) / close      일중 변동폭
  [market — 시장 국면]  (1라운드)
    kospi_ret5, kospi_ret20, rel20 (= ret20 - kospi_ret20, 상대강도)
  [vol — 변동성 국면]  (2라운드, FinRL 터뷸런스 지수의 단일 종목판 + KAIS 2021 의 ATR)
    vol_regime  : std20(ret1) / std250(ret1) - 1   단기 변동성이 1년 평균 대비 얼마나 높은가
    z_ret       : ret1 / std20(ret1)               표준화 일간 수익률 — 오늘이 몇 σ 짜리 날인가
    atr14_ratio : ATR14 / close                    갭 포함 실질 변동폭 (True Range 의 14일 평균)
  [gated — 국면 조건부]  (D-03.9)
    close_ma200_ratio_g, close_hi252_ratio_g : vol_regime > 0(스트레스 국면)일 때만 값, 아니면 0.
  [us — 해외 지수]  (2라운드, KAIS 2021)
    sp_ret1, sp_ret5 : S&P500 1·5일 수익률. 미국 장은 한국 장 마감 뒤 ~ 다음날 새벽에 끝나므로
                       한국 거래일 t 에는 **미국 날짜 ≤ t-1 인 마지막 종가**까지만 쓴다 (룩어헤드 방지).
  [market_vk] (vkospi.csv 있을 때만)  vkospi_ma20_ratio
  [flow] (<code>_flow.csv 있을 때만)  frgn5, frgn20, inst20 — 외국인·기관 순매수금액 / 거래대금
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "out"

STRUCTURE = ["close_ma200_ratio", "close_hi252_ratio", "vol5_vol20", "hl_range"]
MARKET = ["kospi_ret5", "kospi_ret20", "rel20"]
VOL = ["vol_regime", "z_ret", "atr14_ratio"]
GATED = ["close_ma200_ratio_g", "close_hi252_ratio_g"]   # D-03.9: 변동성 국면(vol_regime > 0)일 때만 켜지는 장기 피처
US = ["sp_ret1", "sp_ret5"]
MARKET_VK = ["vkospi_ma20_ratio"]
FLOW = ["frgn5", "frgn20", "inst20"]
ALL_EXTRA = STRUCTURE + MARKET + VOL + GATED + US + MARKET_VK + FLOW


def add_structure(f: pd.DataFrame) -> pd.DataFrame:
    close = f["close"]
    f["close_ma200_ratio"] = close / close.rolling(200).mean() - 1
    f["close_hi252_ratio"] = close / close.rolling(252).max() - 1
    r1 = close.pct_change()
    f["vol5_vol20"] = r1.rolling(5).std() / r1.rolling(20).std() - 1
    f["hl_range"] = (f["high"] - f["low"]) / close
    return f


def add_vol(f: pd.DataFrame) -> pd.DataFrame:
    close = f["close"]
    r1 = close.pct_change()
    sd20 = r1.rolling(20).std()
    f["vol_regime"] = sd20 / r1.rolling(250).std() - 1
    f["z_ret"] = r1 / sd20
    prev_close = close.shift(1)
    tr = pd.concat([f["high"] - f["low"],
                    (f["high"] - prev_close).abs(),
                    (f["low"] - prev_close).abs()], axis=1).max(axis=1)
    f["atr14_ratio"] = tr.rolling(14).mean() / close
    return f


def add_gated(f: pd.DataFrame) -> pd.DataFrame:
    """국면 조건부 피처: 20일 변동성이 1년 평균보다 높을 때(스트레스 국면)만 장기 국면 피처를 보여준다.
    시그널(vol_regime)은 그날까지의 데이터로만 계산되므로 룩어헤드 없음. 임계값 0 은 '1년 평균'이라 튜닝할 숫자가 없다."""
    gate = (f["vol_regime"] > 0).astype(np.float32)
    f["close_ma200_ratio_g"] = f["close_ma200_ratio"] * gate
    f["close_hi252_ratio_g"] = f["close_hi252_ratio"] * gate
    return f


def add_market(f: pd.DataFrame, kospi: pd.Series) -> pd.DataFrame:
    k = kospi.reindex(f.index)                      # 같은 KRX 거래일 — 빠지면 NaN 으로 남겨 검사에서 드러나게
    f["kospi_ret5"] = k.pct_change(5)
    f["kospi_ret20"] = k.pct_change(20)
    f["rel20"] = f["close"].pct_change(20) - f["kospi_ret20"]
    return f


def add_us(f: pd.DataFrame, sp500: pd.Series) -> pd.DataFrame:
    """미국 지수 수익률을 한국 거래일에 t-1 정렬로 붙인다.

    미국 날짜 d 의 종가는 한국 시간 d+1 새벽에 확정 → 한국 거래일 t 에서 쓸 수 있는 마지막 미국 종가는 날짜 ≤ t-1.
    구현: 미국 시계열의 날짜를 하루 뒤로 밀어(d → d+1) '사용 가능 시점'으로 만든 뒤, 한국 거래일에 직전 값으로 채운다.
    """
    us = sp500.sort_index()
    ret = pd.DataFrame({"sp_ret1": us.pct_change(1), "sp_ret5": us.pct_change(5)})
    ret.index = ret.index + pd.Timedelta(days=1)   # 사용 가능 시점
    ret = ret[~ret.index.duplicated(keep="last")]
    aligned = ret.reindex(ret.index.union(f.index)).ffill().reindex(f.index)
    f["sp_ret1"] = aligned["sp_ret1"]
    f["sp_ret5"] = aligned["sp_ret5"]
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


def load_series(name: str, col: str) -> pd.Series | None:
    p = OUT / name
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date")[col].sort_index()


def build(code: str, raw: pd.DataFrame, feat: pd.DataFrame, kospi: pd.Series,
          sp500: pd.Series | None, vk: pd.Series | None, flow: pd.DataFrame | None,
          quiet: bool = False) -> pd.DataFrame:
    """feat(base8 프레임)에 확장 피처를 붙인다. raw 는 워밍업이 긴 피처(ma200·hi252·vol250)용 원본 OHLCV."""
    f = feat.drop(columns=[c for c in ALL_EXTRA if c in feat.columns]).copy()   # 재실행 안전
    long = add_gated(add_vol(add_structure(raw.copy())))[STRUCTURE + VOL + GATED]
    f = f.join(long, how="left")
    f = add_market(f, kospi)
    if sp500 is not None:
        f = add_us(f, sp500)
    elif not quiet:
        print("sp500.csv 없음 → slim5_us 세트 미생성")
    if vk is not None:
        f = add_market_vk(f, vk)
    elif not quiet:
        print("vkospi.csv 없음 → market_vk 세트 미생성")
    if flow is not None:
        f = add_flow(f, flow)
    elif not quiet:
        print(f"{code}_flow.csv 없음 → flow 세트 미생성")
    # structure·market·vol 은 필수: NaN 행 제거 (앞부분 워밍업 + 지수 결측일). 선택 세트는 NaN 을 남긴다.
    return f.dropna(subset=STRUCTURE + MARKET + VOL + GATED)


def load_inputs(code: str):
    raw = pd.read_csv(OUT / f"{code}.csv", parse_dates=["date"], index_col="date").sort_index()
    feat = pd.read_csv(OUT / f"{code}_features.csv", parse_dates=["date"], index_col="date").sort_index()
    kospi = load_series("kospi.csv", "kospi")
    sp500 = load_series("sp500.csv", "sp500")
    vk = load_series("vkospi.csv", "vkospi")
    fp = OUT / f"{code}_flow.csv"
    flow = pd.read_csv(fp, parse_dates=["date"], index_col="date").sort_index() if fp.exists() else None
    return raw, feat, kospi, sp500, vk, flow


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
    code = ap.parse_args().code
    f = build(code, *load_inputs(code))
    f.to_csv(OUT / f"{code}_features.csv")
    new = [c for c in ALL_EXTRA if c in f.columns]
    print(f"\n{len(f)}행 저장 ({f.index[0].date()} ~ {f.index[-1].date()}) → {OUT / (code + '_features.csv')}")
    print(f"추가 컬럼: {new}")
    print(f[new].describe().T[["mean", "std", "min", "max"]].round(4))
    nan = f[new].isna().sum()
    if nan.any():
        print("\nNaN 있는 선택 컬럼 (해당 세트는 그 구간에서 사용 불가):\n", nan[nan > 0])
