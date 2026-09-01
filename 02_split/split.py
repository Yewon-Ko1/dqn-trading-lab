"""02단계: 시간 순 분할 + walk-forward fold + 누수 없는 정규화.

lesson.md 를 먼저 읽는다. # TODO 를 채운 뒤  python 02_split/check_02.py  로 확인.
"""
import pandas as pd


def time_split(df: pd.DataFrame, split_date: str):
    """split_date 이전(<)은 학습, 이후(>=)는 테스트로 나눈다.

    예: time_split(df, "2024-01-01")
        → train: 2024-01-01 전날까지, test: 2024-01-01부터 끝까지
    반환: (train_df, test_df)
    """
    # TODO 1. 날짜 인덱스 비교로 두 구간을 나눠라 (경계일은 test 쪽에만!)
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    return train, test


def make_folds(df: pd.DataFrame, test_years: list[int], train_len: int = 3):
    """walk-forward fold 목록을 만든다.

    test_years=[2021, 2022] 이면:
      fold 1: 학습 2018~2020 (test_year 직전 train_len개 연도), 테스트 2021
      fold 2: 학습 2019~2021,                              테스트 2022
    반환: [(train_df, test_df), ...]

    힌트: 연도 y 에 대해
      train = df.loc[f"{y-train_len}":f"{y-1}"]
      test  = df.loc[f"{y}":f"{y}"]
    """
    folds = []
    for y in test_years:
        # TODO 2. 위 힌트대로 train/test 를 잘라 folds 에 (train, test) 튜플로 추가
        train = df.loc[f"{y-train_len}": f"{y-1}"]
        test = df.loc[f"{y}":f"{y}"]
        folds.append((train, test))
    return folds


def fit_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """cols 컬럼들을 z-score 정규화한다. 통계(평균·표준편차)는 반드시 train에서만 계산.

    반환: (train_norm, test_norm) — cols 만 정규화된 복사본
    """
    train, test = train.copy(), test.copy()

    # TODO 3. train[cols] 의 평균 m 과 표준편차 s 를 구하고,
    #         train[cols] 와 test[cols] 모두 (값 - m) / s 로 바꿔라
    m = train[cols].mean()
    s = train[cols].std()
    train[cols] = (train[cols]-m)/s
    test[cols] = (test[cols]-m/s)
    return train, test


if __name__ == "__main__":
    from pathlib import Path
    feat = pd.read_csv(Path(__file__).resolve().parents[1] / "out" / "005930_features.csv",
                       parse_dates=["date"], index_col="date")
    tr, te = time_split(feat, "2024-01-01")
    print(f"학습 {tr.index[0].date()} ~ {tr.index[-1].date()} ({len(tr)}일)")
    print(f"테스트 {te.index[0].date()} ~ {te.index[-1].date()} ({len(te)}일)")
    for i, (a, b) in enumerate(make_folds(feat, [2021, 2022, 2023, 2024]), 1):
        print(f"F{i}: 학습 {a.index[0].year}~{a.index[-1].year} ({len(a)}일) → 테스트 {b.index[0].year} ({len(b)}일)")
