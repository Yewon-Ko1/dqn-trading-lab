"""01단계 자동 채점.  python study/01_features/check_01.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features  # noqa: E402

EXPECTED = ["ret1", "ret5", "ret20", "close_ma5_ratio", "close_ma20_ratio",
            "ma20_ma60_ratio", "vol20", "volume_ma20_ratio"]


def synthetic(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(5_000_000, 30_000_000, n).astype(float),
    }, index=idx)


def reference(df):
    c, v = df["close"], df["volume"]
    r = pd.DataFrame(index=df.index)
    r["ret1"] = c.pct_change()
    r["ret5"] = c.pct_change(5)
    r["ret20"] = c.pct_change(20)
    r["close_ma5_ratio"] = c / c.rolling(5).mean() - 1
    r["close_ma20_ratio"] = c / c.rolling(20).mean() - 1
    r["ma20_ma60_ratio"] = c.rolling(20).mean() / c.rolling(60).mean() - 1
    r["vol20"] = r["ret1"].rolling(20).std()
    r["volume_ma20_ratio"] = v / v.rolling(20).mean() - 1
    return r


ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


df = synthetic()
try:
    feat = make_features(df)
except Exception as e:  # noqa: BLE001
    print(f"FAIL make_features 실행 중 오류: {type(e).__name__}: {e}")
    sys.exit(1)

unfilled = [c for c in EXPECTED if c in feat.columns and feat[c].dtype == object]
if unfilled:
    print(f"FAIL 아직 '...' 그대로인 TODO 가 있어요: {unfilled}")
    sys.exit(1)

report("컬럼이 모두 있음", all(c in feat.columns for c in EXPECTED),
       f"빠진 컬럼: {[c for c in EXPECTED if c not in feat.columns]}")
report("NaN 없음", not feat.isna().any().any(), "dropna() 했는지, 임시 ma 컬럼이 남아있는지 확인")
report("길이가 300 - 60 + 1 = 241 (ma60 때문에 앞 59행 제거)", len(feat) == 241,
       f"실제 {len(feat)}행. rolling(60) 의 NaN 이 59행이어야 함")

ref = reference(df).dropna()
for col in EXPECTED:
    if col in feat.columns:
        a, b = feat[col].reindex(ref.index), ref[col]
        report(f"{col} 값 일치", np.allclose(a, b, atol=1e-9, equal_nan=False),
               f"최대 오차 {np.nanmax(np.abs(a - b)):.3g}")

# 룩어헤드 검사: 뒤쪽 100일의 가격을 바꿔도 앞쪽 값은 그대로여야 한다
df2 = df.copy()
df2.iloc[200:, df2.columns.get_loc("close")] *= 3
f1, f2 = make_features(df), make_features(df2)
common = f1.index[f1.index < df.index[200]]
report("미래 데이터가 과거 피처에 영향 없음 (look-ahead 없음)",
       np.allclose(f1.loc[common, EXPECTED], f2.loc[common, EXPECTED]),
       "shift(-n) 이나 center=True 같은 미래 참조가 있는지 확인")

print("\n🎉 01단계 통과! 02단계로." if ok else "\n아직 남았어요. lesson.md 의 pandas 예시를 다시 보세요.")
