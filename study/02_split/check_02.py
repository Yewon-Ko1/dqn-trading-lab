"""02단계 자동 채점.  python 02_split/check_02.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split import time_split, make_folds, fit_transform  # noqa: E402

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

# 합성 데이터: 2018~2024, 값은 날짜와 무관한 난수
rng = np.random.default_rng(0)
idx = pd.bdate_range("2018-01-02", "2024-12-30")
df = pd.DataFrame({"a": rng.normal(0, 1, len(idx)), "b": rng.normal(5, 3, len(idx))}, index=idx)

# ── time_split ──
try:
    tr, te = time_split(df, "2024-01-01")
except Exception as e:  # noqa: BLE001
    print(f"FAIL time_split 실행 오류: {type(e).__name__}: {e}")
    sys.exit(1)
if tr is ... or te is ...:
    print("FAIL 아직 '...' 그대로인 TODO 가 있어요 (time_split)")
    sys.exit(1)
report("time_split: 학습이 전부 경계일 이전", tr.index.max() < pd.Timestamp("2024-01-01"),
       "train 에 2024년 데이터가 섞임")
report("time_split: 테스트가 전부 경계일 이후", te.index.min() >= pd.Timestamp("2024-01-01"),
       "test 에 2023년 이전 데이터가 섞임")
report("time_split: 행 손실/중복 없음", len(tr) + len(te) == len(df),
       f"train {len(tr)} + test {len(te)} ≠ 전체 {len(df)} — 경계 등호를 확인 (한쪽만!)")
report("time_split: 시간 순서 유지(셔플 금지)", tr.index.is_monotonic_increasing and te.index.is_monotonic_increasing)

# ── make_folds ──
folds = make_folds(df, [2021, 2022, 2023, 2024])
report("make_folds: fold 4개", isinstance(folds, list) and len(folds) == 4,
       f"실제 {len(folds) if isinstance(folds, list) else type(folds)}개 — folds.append 했는지 확인")
if isinstance(folds, list) and len(folds) == 4:
    good = True
    for i, y in enumerate([2021, 2022, 2023, 2024]):
        a, b = folds[i]
        good &= (a.index.min().year == y - 3) and (a.index.max().year == y - 1)
        good &= (b.index.min().year == y) and (b.index.max().year == y)
        good &= a.index.max() < b.index.min()   # 학습이 테스트보다 항상 과거
    report("make_folds: 각 fold가 학습 3년 → 테스트 1년, 학습이 항상 과거", good,
           "연도 슬라이스 f'{y-3}':f'{y-1}' 와 f'{y}' 확인")

# ── fit_transform ──
tr, te = time_split(df, "2024-01-01")
trn, ten = fit_transform(tr, te, ["a", "b"])
if trn is ... or isinstance(trn, type(Ellipsis)):
    print("FAIL 아직 '...' 그대로인 TODO 가 있어요 (fit_transform)")
    sys.exit(1)
report("fit_transform: 학습 구간이 평균 0, 표준편차 1",
       np.allclose(trn[["a", "b"]].mean(), 0, atol=1e-9) and np.allclose(trn[["a", "b"]].std(), 1, atol=1e-9),
       "train 통계로 train 을 변환했는지 확인")
report("fit_transform: 테스트는 정확히 0/1이 아님 (자기 통계로 재지 않음)",
       not np.allclose(ten[["a", "b"]].mean(), 0, atol=1e-6),
       "test 를 test 자신의 평균으로 정규화하면 안 됨 — train 의 m, s 를 써야 함")
report("fit_transform: 원본을 훼손하지 않음", not np.allclose(tr["a"].values, trn["a"].values),
       "copy 후 변환한 값을 반환해야 함")

# 누수 검사: 테스트 값을 바꿔도 학습 정규화 결과는 그대로여야 한다
te2 = te.copy(); te2["a"] = te2["a"] * 100 + 7
trn2, _ = fit_transform(tr, te2, ["a", "b"])
report("누수 없음: 테스트를 바꿔도 학습 정규화 결과 불변",
       np.allclose(trn["a"].values, trn2["a"].values),
       "통계를 train+test 합쳐서 구하고 있음 — train 에서만!")

print("\n🎉 02단계 통과! 03(기준선)으로." if ok else "\n아직 남았어요. lesson.md 3절을 다시 보세요.")
