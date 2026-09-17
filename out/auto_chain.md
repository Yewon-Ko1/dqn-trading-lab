
# 무인 체인 시작 09-16 08:37  시드 1~20  폴드 [2018, 2019, 2020]
시작 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 3, 'churn': 'hold10', 'penalty': 0.0}

## D-03.7  (09-16 08:37)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'train_len': 3} | -9.2% | 11.1% | -1.0% -17.8% -8.8% | 29 | 0 | 기준 |
| {'train_len': 5} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | (a)+3.7%✓ (b)3/3✓ (c)✓ |
| {'train_len': 0} | -11.9% | 13.6% | +1.9% -16.3% -21.2% | 28 | 0 | (a)-2.7%✗ (b)2/3✓ (c)✓ |
→ 통과 1개, 1위 -5.5%, 3%p 이내 1개 → 단순한 쪽 **{'train_len': 5}** 채택
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-03.6  (09-16 09:33)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'churn': 'hold10', 'penalty': 0.0} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | 기준 |
| {'churn': 'hold10', 'penalty': 0.003} | -8.6% | 14.4% | +7.5% -20.7% -12.8% | 26 | 0 | (a)-3.1%✗ (b)1/3✗ (c)✓ |
| {'churn': 'hold10', 'penalty': 0.01} | -10.9% | 15.0% | +2.8% -20.3% -15.2% | 22 | 0 | (a)-5.4%✗ (b)1/3✗ (c)✓ |
| {'churn': 'none', 'penalty': 0.0} | -25.2% | 12.9% | -4.3% -35.9% -35.4% | 66 | 0 | (a)-19.7%✗ (b)0/3✗ (c)✗ |
| {'churn': 'none', 'penalty': 0.003} | -21.0% | 10.7% | -1.7% -34.8% -26.5% | 52 | 0 | (a)-15.5%✗ (b)0/3✗ (c)✗ |
| {'churn': 'none', 'penalty': 0.01} | -14.6% | 14.4% | +7.5% -33.9% -17.4% | 25 | 0 | (a)-9.1%✗ (b)1/3✗ (c)✓ |
→ 통과 후보 없음 → **기본값 유지** {'churn': 'hold10', 'penalty': 0.0}
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-04  (09-16 11:58)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'hidden': 32} | -9.6% | 10.7% | -0.5% -15.4% -12.7% | 27 | 0 | (a)-4.0%✗ (b)1/3✗ (c)✓ |
| {'hidden': 64} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | 기준 |
| {'hidden': 128} | -11.5% | 13.7% | -1.3% -24.8% -8.4% | 27 | 0 | (a)-5.9%✗ (b)0/3✗ (c)✗ |
| {'hidden': 256} | -12.7% | 13.4% | +0.1% -22.4% -15.8% | 27 | 0 | (a)-7.2%✗ (b)0/3✗ (c)✓ |
→ 통과 후보 없음 → **기본값 유지** {'hidden': 64}
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-05  (09-16 13:23)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'lr': 0.0003} | -12.6% | 12.6% | -0.2% -24.4% -13.1% | 30 | 0 | (a)-7.0%✗ (b)0/3✗ (c)✓ |
| {'lr': 0.001} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | 기준 |
| {'lr': 0.003} | -9.2% | 15.1% | +4.6% -18.6% -13.7% | 22 | 0 | (a)-3.7%✗ (b)1/3✗ (c)✓ |
→ 통과 후보 없음 → **기본값 유지** {'lr': 0.001}
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-06  (09-16 14:06)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'gamma': 0.95} | -12.0% | 11.4% | -0.0% -21.5% -14.3% | 29 | 0 | (a)-6.4%✗ (b)0/3✗ (c)✓ |
| {'gamma': 0.99} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | 기준 |
| {'gamma': 0.995} | -9.0% | 11.6% | +3.0% -20.8% -9.3% | 29 | 0 | (a)-3.5%✗ (b)1/3✗ (c)✓ |
→ 통과 후보 없음 → **기본값 유지** {'gamma': 0.99}
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-07  (09-16 14:59)  고정: {'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

# 무인 체인 시작 09-17 04:59  시드 1~20  폴드 [2018, 2019, 2020]
시작 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 3, 'churn': 'hold10', 'penalty': 0.0}

## D-03.7  (09-17 04:59)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'train_len': 3} | -9.2% | 11.1% | -1.0% -17.8% -8.8% | 29 | 0 | 기준 |
| {'train_len': 5} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | (a)+3.7%✓ (b)3/3✓ (c)✓ |
| {'train_len': 0} | -11.9% | 13.6% | +1.9% -16.3% -21.2% | 28 | 0 | (a)-2.7%✗ (b)2/3✓ (c)✓ |
→ 통과 1개, 1위 -5.5%, 3%p 이내 1개 → 단순한 쪽 **{'train_len': 5}** 채택
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-03.6  (09-17 05:28)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'churn': 'hold10', 'penalty': 0.0} | -5.5% | 13.5% | +1.9% -16.3% -2.2% | 28 | 0 | 기준 |
| {'churn': 'hold10', 'penalty': 0.003} | -8.6% | 14.4% | +7.5% -20.7% -12.8% | 26 | 0 | (a)-3.1%✗ (b)1/3✗ (c)✓ |
| {'churn': 'hold10', 'penalty': 0.01} | -10.9% | 15.0% | +2.8% -20.3% -15.2% | 22 | 0 | (a)-5.4%✗ (b)1/3✗ (c)✓ |
| {'churn': 'none', 'penalty': 0.0} | -25.2% | 12.9% | -4.3% -35.9% -35.4% | 66 | 0 | (a)-19.7%✗ (b)0/3✗ (c)✗ |
| {'churn': 'none', 'penalty': 0.003} | -21.0% | 10.7% | -1.7% -34.8% -26.5% | 52 | 0 | (a)-15.5%✗ (b)0/3✗ (c)✗ |
| {'churn': 'none', 'penalty': 0.01} | -14.6% | 14.4% | +7.5% -33.9% -17.4% | 25 | 0 | (a)-9.1%✗ (b)1/3✗ (c)✓ |
→ 통과 후보 없음 → **기본값 유지** {'churn': 'hold10', 'penalty': 0.0}
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-03.9  (09-17 05:29)  고정: {'episodes': 10, 'window': 10, 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

# 무인 체인 시작 09-17 08:03  시드 1~20  폴드 [2018, 2019, 2020]
시작 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 3, 'churn': 'hold10', 'penalty': 0.0}

## D-03.7  (09-17 08:03)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'churn': 'hold10', 'penalty': 0.0}
| 후보 | 평균 초과 | 시드 std | 폴드별 초과 (2018 2019 2020) | 거래/년 | 퇴화 | 규칙 5 |
|---|---|---|---|---|---|---|
| {'train_len': 3} | -9.2% | 11.1% | -1.0% -17.8% -8.8% | 29 | 0 | 기준 |
| {'train_len': 5} | -6.1% | 13.1% | +4.7% -20.7% -2.2% | 28 | 0 | (a)+3.1%✓ (b)2/3✓ (c)✓ |
| {'train_len': 0} | -7.9% | 12.6% | +4.7% -20.7% -7.8% | 29 | 0 | (a)+1.3%✗ (b)2/3✓ (c)✓ |
→ 통과 1개, 1위 -6.1%, 3%p 이내 1개 → 단순한 쪽 **{'train_len': 5}** 채택
확정 설정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5, 'churn': 'hold10', 'penalty': 0.0}

## D-03.6  (09-17 09:41)  고정: {'episodes': 10, 'window': 10, 'features': 'slim5', 'hidden': 64, 'lr': 0.001, 'gamma': 0.99, 'target_every': 500, 'eps_decay': 0.999, 'train_len': 5}
