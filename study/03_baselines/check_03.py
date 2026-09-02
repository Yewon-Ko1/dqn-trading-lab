"""03단계 자동 채점.  python study/03_baselines/check_03.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import cumulative_return, max_drawdown, annual_volatility, sharpe, sortino  # noqa: E402
from baselines import cash_pv, buy_and_hold_pv, fixed_exposure_pv, ma_crossover_pv  # noqa: E402

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

def filled(x, name):
    if x is ...:
        print(f"FAIL 아직 '...' 그대로인 TODO 가 있어요 ({name})")
        sys.exit(1)
    return x

idx4 = pd.bdate_range("2024-01-01", periods=4)

# ── metrics: 손계산 검증 ──
pv = pd.Series([100.0, 120.0, 90.0, 130.0], index=idx4)
report("누적수익률: 100→130 은 +0.30", np.isclose(filled(cumulative_return(pv), "cumulative_return"), 0.30),
       f"실제 {cumulative_return(pv)}")
report("MDD: 고점 120 → 저점 90 은 -0.25", np.isclose(filled(max_drawdown(pv), "max_drawdown"), -0.25),
       f"실제 {max_drawdown(pv)} — '지금까지의 최고' 대비인지 확인 (cummax)")
pv_up = pd.Series([100.0, 110.0, 121.0], index=idx4[:3])
report("MDD: 하락이 없으면 0", np.isclose(max_drawdown(pv_up), 0.0), f"실제 {max_drawdown(pv_up)}")

rng = np.random.default_rng(0)
idx = pd.bdate_range("2020-01-01", periods=500)
pv_r = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500))), index=idx)
d = pv_r.pct_change().dropna()
report("연환산 변동성 = 일간 std × √252",
       np.isclose(filled(annual_volatility(pv_r), "annual_volatility"), d.std() * np.sqrt(252)),
       "√252 를 곱했는지 확인")
report("Sharpe = (일간 평균×252) / 연환산 변동성",
       np.isclose(filled(sharpe(pv_r), "sharpe"), (d.mean() * 252) / (d.std() * np.sqrt(252))),
       "분자·분모 연환산을 각각 확인")
dn = d[d < 0]
report("Sortino 분모는 음수 수익률만의 std",
       np.isclose(filled(sortino(pv_r), "sortino"), (d.mean() * 252) / (dn.std() * np.sqrt(252))),
       "downside 만 골라서 std 냈는지 확인")
report("현금 PV 의 Sharpe 는 0 (0 나누기 방지)", sharpe(pd.Series(100.0, index=idx)) == 0.0,
       "변동성 0 일 때 0.0 반환 처리")

# ── baselines ──
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 500))), index=idx)

c = filled(cash_pv(close), "cash_pv")
report("현금: PV 가 매일 초기자본 그대로", np.allclose(c.values, 10_000_000) and len(c) == len(close))

bh0 = filled(buy_and_hold_pv(close, commission=0.0, tax=0.0), "buy_and_hold_pv")
report("B&H(비용 0): PV 변화율 = 주가 변화율",
       np.isclose(bh0.iloc[-1] / bh0.iloc[0], close.iloc[-1] / close.iloc[0], rtol=1e-4),
       "n = balance // first, PV = cash + n×종가 확인")
bh = buy_and_hold_pv(close)
report("B&H: 수수료를 내면 비용 0 보다 최종 PV 가 작다", bh.iloc[-1] < bh0.iloc[-1])

fe = filled(fixed_exposure_pv(close), "fixed_exposure_pv")
ref = 10_000_000 * (1 + 0.5 * close.pct_change().fillna(0)).cumprod()
report("고정노출 50%: pv_t = pv_{t-1}(1 + 0.5×ret)", np.allclose(fe.values, ref.values),
       "cumprod 누적곱 확인")

ma_pv, trades = ma_crossover_pv(close)
filled(ma_pv if not isinstance(ma_pv, type(...)) else ..., "ma_crossover_pv")
report("MA교차: 거래 횟수가 1 이상", trades >= 1)

# 룩어헤드 검사: 뒤쪽 가격을 바꿔도 앞쪽 PV 는 그대로여야 한다
close2 = close.copy(); close2.iloc[300:] *= 3
ma_pv2, _ = ma_crossover_pv(close2)
report("MA교차 룩어헤드 없음: 미래 가격을 바꿔도 과거 PV 불변",
       np.allclose(ma_pv.iloc[:299].values, ma_pv2.iloc[:299].values),
       "position 에 shift(1) 을 했는지 확인 — 오늘 신호는 내일부터!")

# 신호 당일 반영(shift 누락)을 직접 잡는 검사
sig_today = (close.rolling(20).mean() > close.rolling(60).mean())
ret = close.pct_change().fillna(0)
pv_leak = 10_000_000 * (1 + ret.where(sig_today, 0.0)).cumprod()
report("MA교차: '당일 신호 당일 반영' 버전과 결과가 다름 (shift 가 실제로 적용됨)",
       not np.allclose(ma_pv.values, pv_leak.values),
       "shift(1) 이 빠진 것 같음")

print("\n🎉 03단계 통과! 04(매매 환경)로." if ok else "\n아직 남았어요. lesson.md 2·3절을 다시 보세요.")
