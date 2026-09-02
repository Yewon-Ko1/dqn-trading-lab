"""03단계 (2/2): 기준선 4종. 각 함수는 종가 시리즈를 받아 PV(포트폴리오 가치) 시리즈를 돌려준다.

lesson.md 1·3절을 먼저 읽는다. 채점:  python study/03_baselines/check_03.py
"""
import numpy as np
import pandas as pd

BALANCE = 10_000_000
COMMISSION = 0.00015   # 매수·매도 수수료 0.015%
TAX = 0.0025           # 매도 시 거래세 0.25%


def cash_pv(close: pd.Series, balance: int = BALANCE) -> pd.Series:
    """현금 100%: 아무것도 안 한다. PV는 매일 balance 그대로."""
    # TODO 1.  힌트: pd.Series(balance, index=close.index, dtype=float)
    return pd.Series( index = close.index, data = balance, dtype = float)


def buy_and_hold_pv(close: pd.Series, balance: int = BALANCE,
                    commission: float = COMMISSION, tax: float = TAX) -> pd.Series:
    """Buy & Hold: 첫날 종가에 최대한 매수, 끝까지 보유. 남은 잔돈은 현금.

    마지막 날 청산(매도 비용 차감)은 하지 않는다 — 평가액 기준.
    """
    first = close.iloc[0]
    # TODO 2. 첫날 가격으로 살 수 있는 주식 수 n (수수료 포함 단가로 나눈 몫, int)
    #         힌트: balance // (first * (1 + commission))
    n =balance//(first*(1+commission)) 
    cash = balance - n * first * (1 + commission)
    # TODO 3. 매일의 PV = 현금 + n × 그날 종가
    return pd.Series(index = close.index, data = cash + n *close, dtype = float)


def fixed_exposure_pv(close: pd.Series, w: float = 0.5,
                      balance: int = BALANCE) -> pd.Series:
    """고정 노출: 항상 자산의 w(기본 50%)만 주식에 태운 효과. 마찰 없는 근사.

    매일  pv_t = pv_{t-1} × (1 + w × 그날 주가 수익률)
    힌트: ret = close.pct_change().fillna(0)
          누적곱: (1 + w * ret).cumprod() × balance
    """
    # TODO 4.
    ret = close.pct_change().fillna(0)
    return pd.Series(index = close.index, data = balance * (1 + w * ret).cumprod())


def ma_crossover_pv(close: pd.Series, short: int = 20, long: int = 60,
                    balance: int = BALANCE, commission: float = COMMISSION,
                    tax: float = TAX):
    """MA 교차: MA20 > MA60 이면 보유, 아니면 현금. 신호는 다음 날 반영(룩어헤드 금지!).

    반환: (pv 시리즈, 거래 횟수)

    구현 순서 (한 줄씩 이미 짜여 있고, TODO 는 딱 한 곳):
      1) 신호: MA_short > MA_long  (True=주식 보유)
      2) 포지션: 신호를 하루 미룬 것  ← ★ 오늘 종가로 계산한 신호는 내일부터
      3) 일별 전략 수익률: 포지션이 True 인 날만 주가 수익률
      4) 포지션이 바뀌는 날마다 매매 비용 차감
    """
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    signal = (ma_s > ma_l)

    # TODO 5. 신호를 하루 미뤄 포지션으로 만든다 (첫날 NaN 은 False 로).
    #         힌트: signal.shift(1).fillna(False)
    position = signal.shift(1).fillna(False)

    ret = close.pct_change().fillna(0)
    strat_ret = ret.where(position, 0.0)            # 보유한 날만 수익률 반영

    switches = position.astype(int).diff().fillna(0)  # +1 매수 진입일, -1 청산일
    cost = pd.Series(0.0, index=close.index)
    cost[switches == 1] = commission                # 매수 비용
    cost[switches == -1] = commission + tax         # 매도 비용 + 세금

    pv = balance * ((1 + strat_ret) * (1 - cost)).cumprod()
    trades = int((switches != 0).sum())
    return pv, trades


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import summarize

    feat = pd.read_csv(Path(__file__).resolve().parents[2] / "out" / "005930_features.csv",
                       parse_dates=["date"], index_col="date")
    close = feat["close"].loc["2024"]               # 2024년 한 해로 성적표
    ma_pv, ma_trades = ma_crossover_pv(close)
    rows = [
        summarize("Buy&Hold", buy_and_hold_pv(close)),
        summarize("MA20/60 교차", ma_pv, ma_trades),
        summarize("고정노출 50%", fixed_exposure_pv(close)),
        summarize("현금 100%", cash_pv(close), 0),
    ]
    print(pd.DataFrame(rows).to_string(index=False))
