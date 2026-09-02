"""04단계: 매매 환경 TradingEnv. lesson.md 를 먼저 읽는다 (특히 0절 클래스).

채점:  python study/04_env/check_04.py
"""
import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret5", "ret20", "close_ma5_ratio", "close_ma20_ratio",
            "ma20_ma60_ratio", "vol20", "volume_ma20_ratio"]


class TradingEnv:
    BUY, SELL, HOLD = 0, 1, 2

    def __init__(self, df: pd.DataFrame, window: int = 20,
                 balance: int = 10_000_000,
                 commission: float = 0.00015, tax: float = 0.0025):
        """df: FEATURES 컬럼 + 'close' 컬럼을 가진 DataFrame (01~02단계 산출물)"""
        self.features = df[FEATURES].values.astype(np.float32)  # (일수, 8)
        self.prices = df["close"].values.astype(float)          # 매매에 쓰는 원가격
        self.window = window
        self.init_balance = balance
        self.commission = commission
        self.tax = tax

    # ── 에피소드 시작 ────────────────────────────────────────────
    def reset(self):
        """기억(잔고·보유주식·시계)을 판 시작 상태로 되돌리고 첫 state 를 반환한다."""
        self.t = self.window          # 첫 window 일은 관측 재료라 window 일째부터 시작
        # TODO 1. 나머지 기억 초기화:
        #   self.balance = 초기자본,  self.shares = 0 (보유 주식 수),
        #   self.asset = 초기자본 (현재 총자산),  self.trades = 0 (거래 횟수)
        ...
        return self._obs()

    # ── state 만들기 ────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        """최근 window 일 피처(펼침) + 포지션 3개 = state 벡터."""
        win = self.features[self.t - self.window:self.t].flatten()
        # TODO 2. 포지션 3개를 np.array 로 만들어라 (dtype=np.float32):
        #   [ 주식평가액/총자산,  총자산/초기자본 - 1,  현금/초기자본 ]
        #   주식평가액 = self.shares * self.prices[self.t]
        pos = ...
        return np.concatenate([win, pos])

    # ── 하루 진행 ────────────────────────────────────────────────
    def step(self, action: int):
        """행동 집행 → 하루 전진 → 보상 계산. 반환: (새 state, reward, done)"""
        price = self.prices[self.t]

        if action == self.BUY:
            # TODO 3. 현금으로 살 수 있는 주식 수 n 을 구해 매수하라.
            #   n = int(self.balance // (price * (1 + self.commission)))
            #   n > 0 이면: 현금에서 n×price×(1+수수료) 차감, shares += n, trades += 1
            #   n == 0 이면 아무 일도 하지 않는다 (액션 마스킹)
            ...

        elif action == self.SELL:
            # TODO 4. 보유 주식이 있으면 전량 매도하라.
            #   self.shares > 0 이면: 현금 += shares×price×(1-수수료-거래세),
            #                         shares = 0, trades += 1
            #   0주면 아무 일도 하지 않는다 (액션 마스킹)
            ...

        # (HOLD 는 아무 것도 안 함)

        self.t += 1                                   # 하루 전진
        new_asset = self.balance + self.shares * self.prices[self.t]

        # TODO 5. 보상과 상태 갱신:
        #   reward = 새 자산 / 직전 자산(self.asset) - 1
        #   self.asset = new_asset
        reward = ...
        ...

        done = self.t >= len(self.prices) - 1
        return self._obs(), reward, done
