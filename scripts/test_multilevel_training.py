import torch
from src.compression.trainable_gain_hyperprior import (
    TrainableGainHyperprior,
    rate_distortion_loss,
)

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TrainableGainHyperprior().to(device)
model.train()

optimizer = torch.optim.Adam(
    [model.log_gain], lr=1e-4
)

lambdas = [0.0018, 0.0035, 0.0067, 0.0130]

x = torch.rand(1, 3, 256, 256, device=device)

old_gain = model.log_gain.detach().clone()

optimizer.zero_grad(set_to_none=True)

print("开始四档联合训练测试")

for level, lmbda in enumerate(lambdas):
    output = model(x, level=level)

    loss, bpp, mse = rate_distortion_loss(
        output, x, lmbda=lmbda
    )

    # 分档反向传播，避免同时保留四张计算图
    (loss / len(lambdas)).backward()

    print(
        f"Level={level} | "
        f"lambda={lmbda:.4f} | "
        f"bpp={bpp.item():.4f} | "
        f"MSE={mse.item():.6f} | "
        f"Loss={loss.item():.4f}",
        flush=True
    )

grad = model.log_gain.grad

if grad is None or not torch.isfinite(grad).all():
    raise RuntimeError("Gain 梯度不存在或含非有限值")

for level in range(4):
    grad_norm = grad[level].norm().item()
    print(f"Level={level} 梯度范数={grad_norm:.8f}")

torch.nn.utils.clip_grad_norm_(
    [model.log_gain], max_norm=1.0
)

optimizer.step()

print("\n===== 参数更新检查 =====")
for level in range(4):
    changed = not torch.equal(
        old_gain[level],
        model.log_gain.detach()[level]
    )
    print(f"Level={level} 参数更新：{changed}")

print("四档联合训练单次测试结束")
