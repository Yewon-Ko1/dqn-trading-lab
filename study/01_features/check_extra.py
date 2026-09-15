"""01단계 (추가) 검사: extra_features 의 룩어헤드·결측·스케일·미국 지수 정렬. 빈칸 없음.

실행:  python study/01_features/check_extra.py [--code 005930]

검사 1 (룩어헤드): 모든 입력(원본 OHLCV·KOSPI·S&P500·VKOSPI·수급)을 기준일에서 잘라 다시 계산해도
        기준일 이전 값이 전부 같아야 한다. 미래 데이터가 조금이라도 섞이면 여기서 FAIL.
검사 5 (미국 지수 정렬): 한국 거래일 t 의 sp_ret1 은 미국 날짜 ≤ t-1 인 마지막 거래일의 수익률과 같아야 한다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_features import (ALL_EXTRA, STRUCTURE, MARKET, VOL, US, build, load_inputs)  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out"
ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
code = ap.parse_args().code

raw, feat, kospi, sp500, vk, flow = load_inputs(code)
f = feat                                    # 현재 저장된 파일 (extra_features 실행 결과)
present = [c for c in ALL_EXTRA if c in f.columns]
ok = True

# 1. 룩어헤드: 기준일 이후 입력을 전부 잘라내고 다시 계산 → 기준일 이전 값 동일해야 함
cut = f.index[len(f) // 2]
tr = lambda x: None if x is None else x.loc[:cut]     # noqa: E731
g = build(code, tr(raw), tr(feat), tr(kospi), tr(sp500), tr(vk), tr(flow), quiet=True)
for c in present:
    a, b = f.loc[:cut, c].dropna(), g[c].dropna()
    common = a.index.intersection(b.index)
    same = len(common) > 0 and np.allclose(a.loc[common], b.loc[common], equal_nan=True)
    print(f"[{'OK' if same else 'FAIL'}] 룩어헤드 없음: {c}")
    ok &= same

# 2. 필수 세트 결측
for c in STRUCTURE + MARKET + VOL:
    n = int(f[c].isna().sum()); print(f"[{'OK' if n == 0 else 'FAIL'}] NaN 0개: {c} ({n})"); ok &= n == 0

# 3. 스케일: 비율형이면 |평균| < 1, std 가 0 이 아님 (z_ret 은 표준화 값이라 |평균| < 1 이면 충분)
for c in present:
    m, s = f[c].mean(), f[c].std()
    good = abs(m) < 1 and s > 0
    print(f"[{'OK' if good else 'FAIL'}] 스케일: {c} mean {m:+.4f} std {s:.4f}"); ok &= good

# 4. 정의 확인
print(f"[{'OK' if (f['close_hi252_ratio'] <= 1e-12).all() else 'FAIL'}] close_hi252_ratio ≤ 0")
print(f"[{'OK' if (f['hl_range'] >= 0).all() else 'FAIL'}] hl_range ≥ 0")
print(f"[{'OK' if (f['atr14_ratio'] > 0).all() else 'FAIL'}] atr14_ratio > 0")

# 5. 미국 지수 정렬 (t-1): 무작위 30일을 뽑아 직접 계산한 값과 비교
if sp500 is not None and "sp_ret1" in f.columns:
    us_ret1 = sp500.pct_change(1)
    rng = np.random.default_rng(0)
    bad = 0
    for t in rng.choice(f.index[300:], 30, replace=False):
        last_us = us_ret1.loc[:t - pd.Timedelta(days=1)].dropna()
        expect = last_us.iloc[-1]
        if not np.isclose(f.loc[t, "sp_ret1"], expect, equal_nan=True):
            bad += 1
    print(f"[{'OK' if bad == 0 else 'FAIL'}] 미국 지수 t-1 정렬 (30일 표본 중 불일치 {bad})"); ok &= bad == 0
    n = int(f["sp_ret1"].isna().sum()); print(f"[{'OK' if n == 0 else 'WARN'}] sp_ret1 NaN {n}개")
else:
    print("[SKIP] 미국 지수 없음 (sp500.csv)")

# 6. 폴드 학습 길이: 2018 폴드 학습(2015~2017) 이 3년 확보됐는지
n2015 = len(f.loc["2015":"2015"]); print(f"[{'OK' if n2015 > 240 else 'WARN'}] 2015년 행 수 {n2015} (240 이상이면 연초부터 확보)")

print("\n전부 통과" if ok else "\n실패 항목 있음")
