"""04단계 자동 채점.  python study/04_env/check_04.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_env import TradingEnv, FEATURES  # noqa: E402

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

def make_df(prices):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 0.01, len(prices)) for c in FEATURES})
    df["close"] = prices
    return df

W = 5  # 검사에서는 window 5 로 작게

# ── reset / _obs ──
env = TradingEnv(make_df(np.full(30, 10_000.0)), window=W)
try:
    obs = env.reset()
    _ = len(obs), env.asset, env.shares, env.trades
except Exception as e:  # noqa: BLE001
    print(f"FAIL reset/_obs 실행 오류 — TODO 1·2 를 먼저 채우세요: {type(e).__name__}: {e}")
    sys.exit(1)
report("state 길이 = window×피처수 + 3", len(obs) == W * len(FEATURES) + 3,
       f"실제 {len(obs)} (기대 {W*len(FEATURES)+3})")
report("reset 직후: 자산=초기자본, 주식 0주, 거래 0회",
       env.asset == 10_000_000 and env.shares == 0 and env.trades == 0,
       "TODO 1 의 초기화 확인")
report("reset 직후 포지션 3개 = [0, 0, 1]", np.allclose(obs[-3:], [0.0, 0.0, 1.0]),
       "주식 0주면 보유비율 0, 수익률 0, 잔고비율 1 이어야 함 (TODO 2)")

# ── 법칙 2: 관망만 하면 자산 불변 ──
env.reset()
done = False
rewards = []
while not done:
    _, r, done = env.step(TradingEnv.HOLD)
    rewards.append(r)
report("관망만 하면 자산이 초기자본 그대로", env.asset == 10_000_000 and env.trades == 0)
report("관망만 하면 모든 보상이 0", np.allclose(rewards, 0.0),
       "reward 를 '자산' 변화율로 계산했는지 확인 (가격 변화율이면 안 됨)")

# ── 법칙 4: 손계산 시나리오 (가격 10,000 → 다음날 20,000) ──
prices = np.full(W + 6, 10_000.0); prices[W + 1:] = 20_000.0
env = TradingEnv(make_df(prices), window=W)
env.reset()
_, r1, _ = env.step(TradingEnv.BUY)      # 10,000 원에 매수 → 다음날 20,000
n_expect = int(10_000_000 // (10_000 * 1.00015))     # 999주
cash_expect = 10_000_000 - n_expect * 10_000 * 1.00015
asset_expect = cash_expect + n_expect * 20_000
report("매수 주식 수가 손계산과 일치 (수수료 반영, 몫)", env.shares == n_expect,
       f"실제 {env.shares}주 (기대 {n_expect}주) — // 와 (1+commission) 확인 (TODO 3)")
report("급등 다음날 자산이 손계산과 일치", np.isclose(env.asset, asset_expect),
       f"실제 {env.asset:,.0f} (기대 {asset_expect:,.0f})")
report("그날 보상 = 자산 변화율", np.isclose(r1, asset_expect / 10_000_000 - 1),
       "reward = new_asset/self.asset - 1, 그 다음 self.asset 갱신 (TODO 5 순서)")

_, r2, _ = env.step(TradingEnv.SELL)     # 20,000 원에 전량 매도
cash_after = cash_expect + n_expect * 20_000 * (1 - 0.00015 - 0.0025)
report("매도 후 현금이 손계산과 일치 (수수료+세금)", np.isclose(env.balance, cash_after),
       f"실제 {env.balance:,.0f} (기대 {cash_after:,.0f}) — (1-commission-tax) 확인 (TODO 4)")
report("매도 후 0주, 거래 2회", env.shares == 0 and env.trades == 2)

# ── 액션 마스킹 ──
t0 = env.trades
env.step(TradingEnv.SELL)                # 0주인데 매도 시도
report("0주 매도는 조용히 무시 (거래 수 그대로)", env.trades == t0, "if self.shares > 0 확인 (TODO 4)")

# ── 법칙 1·3: 랜덤 행동에서도 회계가 맞는다 ──
rng = np.random.default_rng(42)
close = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))))
env = TradingEnv(make_df(close.values), window=W)
env.reset()
done, neg, prod = False, False, 1.0
while not done:
    _, r, done = env.step(int(rng.integers(0, 3)))
    prod *= (1 + r)
    neg |= (env.balance < -1e-6) or (env.shares < 0)
report("랜덤 행동에도 잔고·주식이 음수가 되지 않음", not neg)
report("보상 누적(복리) = 최종 자산 (보상이 곧 돈)", np.isclose(prod * 10_000_000, env.asset, rtol=1e-9),
       "reward 계산이나 self.asset 갱신 누락 확인")

print("\n🎉 04단계 통과! 05(랜덤 에이전트 검증) → python study/05_random/check_05.py" if ok
      else "\n아직 남았어요. lesson.md 2·3절을 다시 보세요.")
