"""00단계: 삼성전자(005930) 일봉 데이터를 CSV로 저장한다.

이 파일은 빈칸이 없다. 실행하고, 만들어진 CSV를 엑셀이나 메모장으로 열어 본다.
기존 backend/app/kis_client.py 는 같은 데이터를 한국투자증권 API에서 받아오는데,
토큰 발급·페이지 나누기 때문에 213줄이다. 우리는 공부용이므로 FinanceDataReader 로 5줄에 끝낸다.

실행:  python study/00_data/get_data.py
결과:  out/005930.csv
"""
import argparse
from pathlib import Path

import FinanceDataReader as fdr  # pip install finance-datareader

OUT = Path(__file__).resolve().parents[2] / "out"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--code", default="005930")
code = ap.parse_args().code
df = fdr.DataReader(code, "2015-01-01")   # 기본 삼성전자, --code 069500 은 KODEX200
df = df.rename(columns=str.lower)              # Open→open, Close→close ...
df.index.name = "date"
df = df[["open", "high", "low", "close", "volume"]]

df.to_csv(OUT / f"{code}.csv")
print(df.head())
print(df.tail())
print(f"\n{len(df)}일치 저장 → {OUT / (code + '.csv')}")

# ── 스스로 답해 볼 것 (notes.md 에 적기) ─────────────────────────────
# Q1. 행 하나는 무엇을 뜻하나? 주말/공휴일 행이 있나?
# Q2. close 가 2015년엔 2만원대인데 2018년에 갑자기 5만원대로 바뀌는 구간이 있나? (액면분할)
#     → 이런 '가격 자체'를 모델에 넣으면 왜 문제가 되는지 01단계에서 다룬다.
# Q3. volume 의 단위는? 값의 크기가 close 와 몇 배 차이 나나?
