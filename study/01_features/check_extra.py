"""01단계 (추가) 검사: extra_features 의 룩어헤드·결측·스케일. 빈칸 없음.

실행:  python study/01_features/check_extra.py [--code 005930]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_features import STRUCTURE, MARKET, MARKET_VK, FLOW, add_structure, add_market  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out"
ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
code = ap.parse_args().code
f = pd.read_csv(OUT / f"{code}_features.csv", parse_dates=["date"], index_col="date")
kospi = pd.read_csv(OUT / "kospi.csv", parse_dates=["date"], index_col="date")["kospi"]
ok = True

# 1. 룩어헤드: 미래 행을 잘라내고 다시 계산해도 과거 값이 같아야 한다
cut = f.index[len(f) // 2]
g = f.loc[:cut, ["open", "high", "low", "close", "volume"]].copy()
g = add_market(add_structure(g), kospi)
for c in STRUCTURE + MARKET:
    a, b = f.loc[:cut, c].dropna(), g[c].dropna()
    common = a.index.intersection(b.index)
    same = np.allclose(a.loc[common], b.loc[common], equal_nan=True)
    print(f"[{'OK' if same else 'FAIL'}] 룩어헤드 없음: {c}")
    ok &= same

# 2. 필수 세트 결측
for c in STRUCTURE + MARKET:
    n = int(f[c].isna().sum()); print(f"[{'OK' if n == 0 else 'FAIL'}] NaN 0개: {c} ({n})"); ok &= n == 0

# 3. 스케일: 비율형이면 |평균| < 1, std 가 0 이 아님
for c in [x for x in STRUCTURE + MARKET + MARKET_VK + FLOW if x in f.columns]:
    m, s = f[c].mean(), f[c].std()
    good = abs(m) < 1 and s > 0
    print(f"[{'OK' if good else 'FAIL'}] 스케일: {c} mean {m:+.4f} std {s:.4f}"); ok &= good

# 4. 정의 확인: 52주 고점 대비는 항상 ≤ 0, 일중 변동폭은 항상 ≥ 0
print(f"[{'OK' if (f['close_hi252_ratio'] <= 1e-12).all() else 'FAIL'}] close_hi252_ratio ≤ 0")
print(f"[{'OK' if (f['hl_range'] >= 0).all() else 'FAIL'}] hl_range ≥ 0")

# 5. 폴드 학습 길이: 2018 폴드 학습(2015~2017) 이 3년 확보됐는지
n2015 = len(f.loc["2015":"2015"]); print(f"[{'OK' if n2015 > 240 else 'WARN'}] 2015년 행 수 {n2015} (240 이상이면 연초부터 확보)")

print("\n전부 통과" if ok else "\n실패 항목 있음")
