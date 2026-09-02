"""03단계 (1/2): 성과지표. PV(포트폴리오 가치) 시리즈 하나를 받아 성적표를 만든다.

lesson.md 2절을 먼저 읽는다. 채점:  python study/03_baselines/check_03.py
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # 1년 거래일 수 (연환산에 사용)


def cumulative_return(pv: pd.Series) -> float:
    """누적수익률: 마지막 PV / 처음 PV - 1"""
    return pv.iloc[-1] / pv.iloc[0] - 1


def max_drawdown(pv: pd.Series) -> float:
    """최대 낙폭(MDD): '지금까지의 최고 PV' 대비 낙폭의 최솟값. 항상 0 이하.

    힌트: peak = pv.cummax()  ← 각 시점까지의 최고값 시리즈
          낙폭 = pv / peak - 1
    """
    # TODO 2.
    peak = pv.cummax()
    drowdown = pv / peak -1
    return drowdown.min()


def annual_volatility(pv: pd.Series) -> float:
    """연환산 변동성: 일간 수익률의 표준편차 × √252"""
    daily = pv.pct_change().dropna()
    # TODO 3.  힌트: daily.std() 와 np.sqrt(TRADING_DAYS)
    return daily.std() * np.sqrt(TRADING_DAYS)


def sharpe(pv: pd.Series) -> float:
    """Sharpe = 연환산 수익률 / 연환산 변동성 (무위험 0 가정).

    연환산 수익률은 일간 평균 × 252 로 근사한다.
    변동성이 0이면(거래 없음 등) 0.0 을 반환한다.
    """
    daily = pv.pct_change().dropna()
    vol = annual_volatility(pv)
    if vol == 0 or np.isnan(vol):
        return 0.0
    # TODO 4.  힌트: daily.mean() * TRADING_DAYS 를 vol 로 나눔
    return   daily.mean() * TRADING_DAYS / vol

def sortino(pv: pd.Series) -> float:
    """Sortino = 연환산 수익률 / 하락 변동성(음수 수익률만의 표준편차 × √252).

    음수 수익률이 없거나 하락 변동성이 0이면 0.0 을 반환한다.
    """
    daily = pv.pct_change().dropna()
    downside = daily[daily < 0]
    if len(downside) == 0:
        return 0.0
    # TODO 5.  힌트: 분모가 downside.std() * np.sqrt(TRADING_DAYS) 인 것만 Sharpe 와 다름
    dvol = downside.std() * np.sqrt(TRADING_DAYS)
    if dvol == 0 or np.isnan(dvol):
        return 0.0
    return (daily.mean() * TRADING_DAYS) / dvol


def summarize(name: str, pv: pd.Series, trades: int | None = None) -> dict:
    """한 전략의 성적표 한 줄."""
    return {
        "전략": name,
        "누적수익률": round(cumulative_return(pv), 4),
        "MDD": round(max_drawdown(pv), 4),
        "Sharpe": round(sharpe(pv), 3),
        "Sortino": round(sortino(pv), 3),
        "거래수": trades,
    }
