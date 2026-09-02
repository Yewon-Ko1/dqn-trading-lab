# 01단계 연습장 — 셀 위의 "Run Cell"을 눌러 한 칸씩 실행하며 출력을 눈으로 확인한다.
# 확인이 끝난 코드만 features.py 의 TODO 로 옮긴다. (이 파일은 커밋 안 해도 됨)

# %% 0. 데이터 불러오기 (제일 먼저 이 셀부터)
import pandas as pd
from pathlib import Path

# 실행 위치가 어디든 out/005930.csv 를 찾는다 (repo 루트 / 01_features 둘 다 대응)
csv = next(p for p in [Path("out/005930.csv"), Path("../out/005930.csv")] if p.exists())
df = pd.read_csv(csv, parse_dates=["date"], index_col="date")
df.head()

# %% 1. shift — 한 칸 아래로 밀기. 첫 행이 NaN이 되고, t행에 t-1 값이 온다
df["close"].head(5)

# %%
df["close"].shift(1).head(5)   # 위 셀과 비교: 값이 한 칸씩 내려왔는지 확인

# %% 2. pct_change — (오늘-어제)/어제. 아래 두 셀의 결과가 같아야 한다
df["close"].pct_change().head(5)

# %%
(df["close"] / df["close"].shift(1) - 1).head(5)

# %% 3. rolling — 20일 이동평균. 앞 19개가 NaN인지 확인
df["close"].rolling(20).mean().head(25)

# %% 4. 이격도 — 종가가 20일 평균보다 위(+)인지 아래(-)인지
(df["close"] / df["close"].rolling(20).mean() - 1).tail(10)

# %% 5. 변동성 — 일간 수익률의 20일 표준편차. 2020년 3월(코로나)에 튀는지 확인
ret1 = df["close"].pct_change()
ret1.rolling(20).std().loc["2020-02":"2020-05"]

# %% 6. 자유 실험 칸 — 여기서 마음껏
v = df["close"].pct_change().rolling(20).std()      # vol20 전체
peak_day = v.loc["2020"].idxmax()                   # 2020년 중 최대인 날짜
normal = v.loc["2020-02-03"]                        # 코로나 전 평상시 값
print("최고점 날짜:", peak_day)
print("평상시:", round(normal, 4), "→ 최고점:", round(v.max(), 4),
      "=", round(v.loc["2020"].max() / normal, 1), "배")
# %%
