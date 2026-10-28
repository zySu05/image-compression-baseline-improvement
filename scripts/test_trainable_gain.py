import torch
from src.compression.trainable_gain_hyperprior import (
    TrainableGainHyperprior,
    rate_distortion_loss,
)

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("训练设备：", device)

model = TrainableGainHyperprior().to(device)
model.train()

optimizer = torch.optim.Adam([model.log_gain], lr=1e-4)

# 仅用于测试训练流程的随机输入
x = torch.rand(1, 3, 256, 256, device=device)

optimizer.zero_grad(set_to_none=True)

output = model(x, level=1)
loss, bpp, mse = rate_distortion_loss(
    output, x, lmbda=0.01
)

loss.backward()

grad = model.log_gain.grad
print("总损失：", loss.item())
print("估计bpp：", bpp.item())
print("MSE：", mse.item())
print("Gain梯度存在：", grad is not None)
print("Gain梯度有限：",
      grad is not None and torch.isfinite(grad).all().item())
print("Gain梯度非零：",
      grad is not None and grad.abs().sum().item() > 0)

old = model.log_gain.detach().clone()
optimizer.step()

print("Gain参数已更新：",
      not torch.equal(old, model.log_gain.detach()))
print("输出形状：", tuple(output["x_hat"].shape))
print("完整模型单次训练测试结束")
