"""06단계 자동 채점.  python study/06_network/check_06.py"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qnet import build_qnet, train_step, fit_toy  # noqa: E402

ok = True
def report(name, cond, hint=""):
    global ok
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)

# ── 구조 ──
try:
    model = build_qnet(163)
    out = model(torch.randn(5, 163))
except Exception as e:  # noqa: BLE001
    print(f"FAIL build_qnet 실행 오류 — TODO 1 을 먼저 채우세요: {type(e).__name__}: {e}")
    sys.exit(1)
report("출력 shape 이 (배치 5, 행동 3)", tuple(out.shape) == (5, 3), f"실제 {tuple(out.shape)}")
last = list(model.children())[-1]
report("마지막 층이 Linear (ReLU 아님)", isinstance(last, nn.Linear),
       "마지막 층 뒤 ReLU 제거 — Q값은 음수 가능해야 함")
n_relu = sum(isinstance(m, nn.ReLU) for m in model.children())
report("숨은 층 사이에 ReLU 2개", n_relu == 2, f"실제 {n_relu}개")
report("Q값이 음수도 낼 수 있음", (build_qnet(10)(torch.randn(200, 10)) < 0).any().item(),
       "마지막 ReLU 가 남아있는 것 같음")

# ── 4박자 ──
torch.manual_seed(1)
m = build_qnet(10)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
before = [p.clone() for p in m.parameters()]
try:
    loss1 = train_step(m, opt, torch.randn(32, 10), torch.randn(32, 3))
except Exception as e:  # noqa: BLE001
    print(f"FAIL train_step 실행 오류 — TODO 2 를 먼저 채우세요: {type(e).__name__}: {e}")
    sys.exit(1)
changed = any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
report("step 후 파라미터가 실제로 움직임", changed, "backward() 와 step() 이 둘 다 있는지 확인")
report("loss 가 숫자(float)로 반환됨", isinstance(loss1, float))

# zero_grad 누락 검사: 같은 배치로 여러 번 학습해도 발산하지 않아야
torch.manual_seed(2)
m2 = build_qnet(10)
opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
xb, yb = torch.randn(32, 10), torch.randn(32, 3)
ls = [train_step(m2, opt2, xb, yb) for _ in range(50)]
report("같은 배치 반복 학습 시 loss 감소 (zero_grad 정상)", ls[-1] < ls[0],
       f"{ls[0]:.4f} → {ls[-1]:.4f} — optimizer.zero_grad() 누락이면 발산함")

# ── 장난감 학습 ──
first, last_loss, _ = fit_toy(seed=0)
report("장난감 문제에서 loss 가 크게 감소 (최소 5배)", last_loss < first / 5,
       f"{first:.4f} → {last_loss:.4f}")
f2, l2, _ = fit_toy(seed=0)
report("같은 시드 = 같은 결과 (재현성)", abs(l2 - last_loss) < 1e-12,
       "torch.manual_seed 이후 난수 사용 순서가 달라졌는지 확인")

print("\n🎉 06단계 통과! 다음은 07 — 표준 DQN." if ok else "\n아직 남았어요. lesson.md 3절(4박자)을 다시 보세요.")
