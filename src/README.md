# src

현재 실행 가능한 기준 코드는 `study/`에 있습니다. 각 구현 옆에 개념 설명(`lesson.md`)과 검증 스크립트(`check_XX.py`)를 함께 두어, 결과뿐 아니라 구현·검증 과정을 추적할 수 있게 했습니다.

핵심 실행 경로는 다음과 같습니다.

- 피처: `study/01_features/`
- 매매 환경: `study/04_env/trading_env.py`
- DQN·Double DQN: `study/07_dqn/dqn.py`
- 본실험: `study/08_runner/run_experiment.py`
- 튜닝: `study/09_tuning/tune.py`, `auto_chain.py`

최종 설정과 외부 연동 인터페이스가 동결되면 재사용 모듈만 이 디렉터리로 분리합니다. 실험 도중 코드를 복제해 두 구현이 달라지는 것을 피하기 위해 현재는 단일 구현을 유지합니다.
