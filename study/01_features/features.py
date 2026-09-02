"""01단계: OHLCV → 피처 테이블.

lesson.md 를 먼저 읽는다. # TODO 를 채운 뒤  python study/01_features/check_01.py  로 확인.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "out"


def load_ohlcv(path: Path) -> pd.DataFrame:
    """CSV 를 읽어 date 를 인덱스로, 날짜순 정렬된 DataFrame 을 돌려준다."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame 에 피처 8개를 붙여서 돌려준다.

    반환 컬럼 (순서 무관):
        open, high, low, close, volume,
        ret1, ret5, ret20, close_ma20_ratio, ma20_ma60_ratio,
        vol20, volume_ma20_ratio, close_ma5_ratio
    NaN 이 생기는 앞부분 행은 제거한다.
    """
    out = df.copy()
    close = out["close"]

    # TODO 1. 일간 수익률: 오늘 종가 / 어제 종가 - 1
    out["ret1"] = close .pct_change(1)

    # TODO 2. 5일, 20일 수익률
    out["ret5"] = close.pct_change(5)
    out["ret20"] = close.pct_change(20)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # TODO 4. 이격도: 종가 / 이동평균 - 1
    out["close_ma5_ratio"] = close/ma5 -1
    out["close_ma20_ratio"] =close/ma20-1
    out["ma20_ma60_ratio"] = ma20/ma60-1

    # TODO 5. 변동성: ret1 의 20일 표준편차
    out["vol20"] = out["ret1"].rolling(20).std()


    # TODO 6. 거래량 이격도: volume / volume 20일 평균 - 1
    out["volume_ma20_ratio"] = out["volume"]/out["volume"].rolling(20).mean()-1

    return out.dropna()


if __name__ == "__main__":
    df = load_ohlcv(OUT / "005930.csv")
    feat = make_features(df)
    feat.to_csv(OUT / "005930_features.csv")
    print(feat.tail())
    print(f"\n{len(df)}행 → {len(feat)}행 (앞부분 NaN 제거됨). 컬럼: {list(feat.columns)}")
    # 값의 크기가 전부 0 근처인지 눈으로 확인:
    print(feat.drop(columns=["open", "high", "low", "close", "volume"]).describe().T[["mean", "std", "min", "max"]])
