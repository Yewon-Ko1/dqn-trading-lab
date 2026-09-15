"""05단계: 랜덤 에이전트로 환경 검증. 빈칸 없음 — 04를 통과한 환경으로 실행만 한다.

  python study/05_random/check_05.py

왜 하나: 환경이 올바르면 '아무렇게나 행동하는' 에이전트는
  ① 대박도 쪽박도 아닌, 대략 본전 근처 ± 시장 노이즈
  ② 단, 매매를 자주 하므로 거래비용만큼은 평균적으로 잃어야
정상이다. 랜덤이 돈을 잘 벌면 환경 어딘가에 공짜 점심(버그)이 있다는 뜻이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_env"))
from trading_env import TradingEnv, FEATURES  # noqa: E402

rng = np.random.default_rng(7)
n_days, n_episodes = 250, 100

finals, trades = [], []
for ep in range(n_episodes):
    # 에피소드마다 새 합성 주가 (추세 없음: 일간 수익률 평균 0)
    # → 랜덤 정책의 기대 성적은 '거래비용만큼만 마이너스'가 되어야 정상
    r = rng.normal(0, 0.015, n_days)
    prices = 10_000 * np.cumprod(1 + r)
    df = pd.DataFrame({c: rng.normal(0, 0.01, n_days) for c in FEATURES})
    df["close"] = prices
    env = TradingEnv(df, window=20)
    env.reset()
    done = False
    while not done:
        _, _, done = env.step(int(rng.integers(0, 3)))
    finals.append(env.asset)
    trades.append(env.trades)

finals = np.array(finals)
ret = finals / 10_000_000 - 1
print(f"에피소드 {n_episodes}회 | 평균 수익률 {ret.mean():+.2%} (표준편차 {ret.std():.2%}) "
      f"| 평균 거래 {np.mean(trades):.0f}회")

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

report("랜덤 정책의 평균 수익률이 0 이하 (공짜 점심 없음)", ret.mean() <= 0.0,
       "랜덤이 돈을 벌면 환경에 버그(룩어헤드·회계 오류)가 있다")
report("그렇다고 전멸도 아님 (평균 -20% 이내)", ret.mean() > -0.20,
       "비용이 이중 차감되는지 확인")
report("거래가 실제로 일어남 (평균 10회 이상)", np.mean(trades) >= 10)
report("파산(자산 0 이하) 없음", finals.min() > 0)

print("\n🎉 05단계 통과! 다음은 06(PyTorch 신경망)." if ok else "\n04 환경을 다시 점검하세요.")
