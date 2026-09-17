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
                 commission: float = 0.00015, tax: float = 0.0025,
                 features: list | None = None,
                 extra_state: bool = False, min_hold: int = 0,
                 trade_penalty: float = 0.0):
        """df: FEATURES 컬럼 + 'close' 컬럼을 가진 DataFrame (01~02단계 산출물)

        features: 사용할 피처 컬럼 목록. None 이면 기본 FEATURES 8종 (기존 동작 그대로).
        09단계 튜닝에서 피처 세트를 바꿔 비교할 때만 지정한다.

        D-03.5 (잦은 매매) 옵션 — 기본값이면 기존 동작과 완전히 같다:
          extra_state: True 면 상태 벡터 끝에 2개 추가 → [보유일수/20, 진입가 대비 손익률].
                       에이전트가 "방금 샀다"를 보고 되팔기 비용을 스스로 배우게 하는 부드러운 방식.
          min_hold:    매수 후 이 거래일 수가 지나기 전에는 SELL 을 HOLD 로 처리(액션 마스킹).
                       되팔기를 막는 딱딱한 방식. 0 이면 제약 없음.
          trade_penalty: (D-03.6) 매매가 실제로 체결된 스텝의 보상에서 이 값을 뺀다. 자산·수익률에는 영향 없음 —
                       학습 신호에만 비용을 과장해 넣어 에이전트가 되팔기 비용을 스스로 배우게 하는 부드러운 방식.
                       0 이면 기존 보상 그대로. (실제 비용 0.265%/왕복은 하루 변동 2% 에 묻혀 신호가 되지 못한다는 가설)
        """
        cols = FEATURES if features is None else list(features)
        self.features = df[cols].values.astype(np.float32)      # (일수, 피처수)
        self.prices = df["close"].values.astype(float)          # 매매에 쓰는 원가격
        self.window = window
        self.init_balance = balance
        self.commission = commission
        self.tax = tax
        self.extra_state = extra_state
        self.min_hold = min_hold
        self.trade_penalty = trade_penalty

    # ── 에피소드 시작 ────────────────────────────────────────────
    def reset(self):
        """기억(잔고·보유주식·시계)을 판 시작 상태로 되돌리고 첫 state 를 반환한다."""
        self.t = self.window          # 첫 window 일은 관측 재료라 window 일째부터 시작
        # TODO 1. 나머지 기억 초기화:
        #   self.balance = 초기자본,  self.shares = 0 (보유 주식 수),
        #   self.asset = 초기자본 (현재 총자산),  self.trades = 0 (거래 횟수)
        self.balance = self.init_balance;
        self.shares = 0;
        self.asset = self.init_balance
        self.trades = 0;
        self.hold_days = 0            # 현재 포지션 보유 거래일 수 (D-03.5)
        self.entry_price = 0.0        # 현재 포지션 진입가 (D-03.5)
        return self._obs()

    # ── state 만들기 ────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        """최근 window 일 피처(펼침) + 포지션 3개 = state 벡터."""
        win = self.features[self.t - self.window:self.t].flatten()
        # TODO 2. 포지션 3개를 np.array 로 만들어라 (dtype=np.float32):
        #   [ 주식평가액/총자산,  총자산/초기자본 - 1,  현금/초기자본 ]
        #   주식평가액 = self.shares * self.prices[self.t]
        pos = np.array([self.shares * self.prices[self.t] / self.asset, 
                        self.asset / self.init_balance - 1, 
                        self.balance / self.init_balance], dtype=np.float32)
        if self.extra_state:                                   # D-03.5: 자기 상태 2개
            pnl = self.prices[self.t] / self.entry_price - 1 if self.shares > 0 else 0.0
            pos = np.concatenate([pos, np.array([self.hold_days / 20.0, pnl], dtype=np.float32)])
        return np.concatenate([win, pos])

    # ── 하루 진행 ────────────────────────────────────────────────
    def step(self, action: int):
        """행동 집행 → 하루 전진 → 보상 계산. 반환: (새 state, reward, done)"""
        price = self.prices[self.t]
        trades_before = self.trades

        if action == self.SELL and self.shares > 0 and self.hold_days < self.min_hold:
            action = self.HOLD                                 # D-03.5: 최소 보유기간 마스킹

        if action == self.BUY:
            # TODO 3. 현금으로 살 수 있는 주식 수 n 을 구해 매수하라.
            #   n = int(self.balance // (price * (1 + self.commission)))
            #   n > 0 이면: 현금에서 n×price×(1+수수료) 차감, shares += n, trades += 1
            #   n == 0 이면 아무 일도 하지 않는다 (액션 마스킹)
            n = int(self.balance // (price * (1 + self.commission)))
            if n > 0:
                if self.shares == 0:                           # 새 포지션 진입 (D-03.5 추적)
                    self.entry_price, self.hold_days = price, 0
                self.balance -= n * price * ( 1 + self.commission)
                self.shares += n
                self.trades += 1
            elif n == 0:
                pass


        elif action == self.SELL:
            # TODO 4. 보유 주식이 있으면 전량 매도하라.
            #   self.shares > 0 이면: 현금 += shares×price×(1-수수료-거래세),
            #                         shares = 0, trades += 1
            #   0주면 아무 일도 하지 않는다 (액션 마스킹)
            if self.shares > 0:
                self.balance += self.shares * price * (1 - self.commission - self.tax)
                self.shares =0 
                self. trades += 1
                self.hold_days, self.entry_price = 0, 0.0

        # (HOLD 는 아무 것도 안 함)

        self.t += 1                                   # 하루 전진
        if self.shares > 0:
            self.hold_days += 1
        new_asset = self.balance + self.shares * self.prices[self.t]

        # TODO 5. 보상과 상태 갱신:
        #   reward = 새 자산 / 직전 자산(self.asset) - 1
        #   self.asset = new_asset
        reward = new_asset / self.asset - 1
        if self.trades > trades_before:                        # D-03.6: 체결된 스텝에만 패널티
            reward -= self.trade_penalty
        self.asset = new_asset

        done = self.t >= len(self.prices) - 1
        return self._obs(), reward, done
