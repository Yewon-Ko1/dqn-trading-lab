# KAIRO 연결 메모

`Gyuho-Han/99-_web`의 KAIRO 백엔드는 `backend/app/agent/policy.py`에서 정책을 등록합니다.
현재 저장소의 DQN은 같은 `buy/sell/hold` 3개 행동을 쓰므로 연결 가능합니다.

## 1. 현재 DQN checkpoint 저장

```bash
python study/07_dqn/train.py --episodes 10 --seed 1 --checkpoint models/dqn_policy.pt
```

생성되는 `models/dqn_policy.pt`는 `.gitignore`에 의해 커밋되지 않습니다.

## 2. KAIRO 백엔드에 정책 등록

`dqn_lab_policy.py`를 KAIRO의 `backend/app/agent/` 아래에 두고,
KAIRO `backend/app/agent/policy.py` 맨 아래에서 기본 정책 등록 이후에 다음을 추가합니다.

```python
from app.agent.dqn_lab_policy import register_dqn_lab_policy

register_dqn_lab_policy("dqn-lab-v1", "models/dqn_policy.pt")
```

KAIRO 서버를 다시 띄우면 에이전트 화면의 정책 선택 목록에 `dqn-lab-v1`이 나타납니다.

## 주의

현재 DQN의 포지션 상태 3개는 학습 환경과 KAIRO 실거래 Observation이 완전히 같지 않습니다.
기술적으로는 연결되지만, 실거래 또는 모의투자 성능 검증 전에는 연구용/시뮬레이션용으로만
다루는 것이 맞습니다.
