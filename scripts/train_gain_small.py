import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor

from src.compression.trainable_gain_hyperprior import (
    TrainableGainHyperprior,
    rate_distortion_loss,
)

torch.manual_seed(42)
random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

images = sorted(
    Path("data/train/div2k").rglob("*.png")
)[:16]

if len(images) < 16:
    raise RuntimeError("DIV2K训练图像不足16张")

checkpoint_dir = Path("checkpoints/gain")
result_dir = Path("results/training")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)

model = TrainableGainHyperprior().to(device)
model.train()

# 只优化增益参数
optimizer = torch.optim.Adam(
    [model.log_gain],
    lr=1e-4
)

lambdas = [0.0018, 0.0035, 0.0067, 0.0130]

initial_gain = model.log_gain.detach().clone()
history = []

print("训练设备:", device)
print("训练图像数量:", len(images))
print("开始小规模四档联合训练", flush=True)

for step in range(1, 9):
    path = random.choice(images)

    with Image.open(path) as im:
        crop = RandomCrop(256)(im.convert("RGB"))
        x = to_tensor(crop).unsqueeze(0).to(device)

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_bpp = 0.0

    for level, lmbda in enumerate(lambdas):
        output = model(x, level)

        loss, bpp, mse = rate_distortion_loss(
            output, x, lmbda
        )

        (loss / len(lambdas)).backward()

        total_loss += loss.detach().item() / len(lambdas)
        total_bpp += bpp.detach().item() / len(lambdas)

    torch.nn.utils.clip_grad_norm_(
        [model.log_gain], max_norm=1.0
    )

    optimizer.step()

    if not torch.isfinite(model.log_gain).all():
        raise RuntimeError("Gain参数出现非有限值")

    history.append({
        "step": step,
        "loss": total_loss,
        "estimated_bpp": total_bpp
    })

    print(
        f"Step {step:02d}/08 | "
        f"Loss={total_loss:.5f} | "
        f"Estimated bpp={total_bpp:.5f}",
        flush=True
    )

checkpoint = checkpoint_dir / "gain_small_trained.pth"

torch.save({
    "step": 8,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "lambdas": lambdas,
}, checkpoint)

csv_path = result_dir / "gain_small_history.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["step", "loss", "estimated_bpp"]
    )
    writer.writeheader()
    writer.writerows(history)

print("\n===== 训练检查 =====")

for level in range(4):
    changed = not torch.equal(
        initial_gain[level],
        model.log_gain.detach()[level]
    )
    print(f"Level={level} 参数更新: {changed}")

print("Checkpoint:", checkpoint)
print("训练记录:", csv_path)
print("小规模真实图像训练完成")
