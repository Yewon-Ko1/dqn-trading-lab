# 01. 피처(feature) — 모델에게 무엇을 보여줄 것인가

> **rltrader에서는:** `data_manager.py`의 `preprocess()`가 만드는 `COLUMNS_TRAINING_DATA_V1`이 바로 이것이다.
> `close_ma20_ratio = (close − ma20) / ma20` 같은 공식이 그대로다. 그 함수는 `rolling`과 `.values[1:]` 슬라이싱으로 짰는데,
> 여기서는 같은 결과를 `shift` / `pct_change` / `rolling`으로 짧게 짜 본다. 이미 아는 내용이니 하루 안에 통과하는 것이 목표.
> 팀 코드 `backend/app/features.py`의 `V1_COLUMNS`도 이름까지 같다 — 팀 코드, rltrader, 내 코드 셋이 같은 값을 내야 한다.

## 왜 가격을 그대로 넣으면 안 되나

신경망은 입력 숫자의 **크기** 에 민감하다. 삼성전자 종가는 5만~8만, 거래량은 1천만~3천만.
이 둘을 한 벡터에 넣으면 거래량이 모든 걸 덮어 버린다. 그리고 "종가 70,000" 이라는 숫자는
그 자체로는 아무 뜻이 없다. 2020년의 70,000과 2024년의 70,000은 완전히 다른 상황이다.

그래서 트레이딩에서 피처는 거의 항상 **비율** 로 만든다.

| 이름 | 계산 | 뜻 |
|---|---|---|
| `ret1` | close[t] / close[t-1] − 1 | 오늘 수익률. "어제보다 몇 % 올랐나" |
| `ret5`, `ret20` | close[t] / close[t-5] − 1 | 1주·1달 수익률 |
| `ma5`, `ma20`, `ma60` | 최근 N일 종가 평균 | 이동평균(추세선) |
| `close_ma20_ratio` | close[t] / ma20[t] − 1 | **이격도**: 지금 가격이 추세선보다 위(+)냐 아래(−)냐 |
| `ma20_ma60_ratio` | ma20 / ma60 − 1 | 단기 추세가 장기 추세 위인가 (골든/데드크로스) |
| `vol20` | ret1 의 최근 20일 표준편차 | 변동성. 요즘 시장이 얼마나 출렁이나 |
| `volume_ma20_ratio` | volume / volume의 20일 평균 − 1 | 평소보다 거래가 몰렸나 |

전부 "−1" 을 붙여서 **0 근처에 모이게** 만든다. 이게 정규화의 첫걸음이다.
기존 `backend/app/features.py` 의 `V1_COLUMNS` 15개와 `MARKET_STATE_COLUMNS` 8개가 정확히
이런 것들이다. 이름만 다르다 (`close_ma5_ratio` 등). 우리는 8개만 만든다.

## 시간 축에서 절대 하면 안 되는 것: 미래 보기 (look-ahead)

`ma20[t]` 는 t일 **까지의** 20일 평균이어야 한다. t일 기준으로 "앞으로 20일" 을 쓰면
백테스트 수익률은 환상적으로 나오고, 실전에서는 쓰레기가 된다. pandas 의 `rolling()` 은
기본적으로 과거 방향이라 안전하지만, `shift(-1)` 처럼 음수 shift 는 미래를 당겨오는 것이니 주의.

## pandas 에서 이걸 하는 방법 (이번 단계에서 익힐 것)

```python
df["close"].shift(1)                 # 한 칸 아래로 밀기 → t행에 t-1 값이 옴
df["close"].pct_change()             # close[t]/close[t-1] - 1 을 한 번에
df["close"].pct_change(5)            # 5일 전 대비
df["close"].rolling(20).mean()       # 20일 이동평균 (앞 19행은 NaN)
df["close"].rolling(20).std()        # 20일 표준편차
df.dropna()                          # NaN 있는 행 제거
```



직접 파이썬 셸이나 Jupyter에서 `df["close"].shift(1).head(5)` 같은 걸 찍어 보면서
**출력을 눈으로 확인** 하는 습관이 이번 단계의 진짜 목표다.

## 과제

1. `features.py` 의 `# TODO` 6곳을 채운다.
2. `python 01_features/check_01.py` 가 전부 `OK` 를 출력하면 통과.
3. `out/005930_features.csv` 를 열어서, 2020년 3월(코로나 급락) 근처의 `ret20`, `vol20`,
   `close_ma20_ratio` 값이 어떻게 생겼는지 `notes.md` 에 두세 줄 적는다.
4. 기존 `backend/app/features.py` 를 열어 `close_ma20_ratio` 를 어디서 어떻게 계산하는지 찾아
   내 코드와 비교한다. (힌트: 함수 이름에 `v1` 이 들어감)

## 자주 하는 실수

- `rolling(20).mean()` 결과의 앞 19행이 NaN 인 걸 잊고 그대로 학습에 넣음 → `dropna()` 또는 warm-up 구간 필요.
- `pct_change()` 를 `close` 가 아니라 이미 비율인 컬럼에 또 적용.
- `volume` 이 0인 날(거래정지)에 나누기 → inf. 이번 단계에선 삼성전자라 없지만, 나중에 다른 종목에서 터진다.
