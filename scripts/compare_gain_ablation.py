import csv
import math
from pathlib import Path

import torch
import lpips
from PIL import Image
from torchvision.transforms.functional import to_tensor
from pytorch_msssim import ms_ssim

from src.compression.trainable_gain_hyperprior import (
    TrainableGainHyperprior,
)

torch.set_num_threads(4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/kodak")
OUT_DIR = Path("results/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "before": Path("checkpoints/gain/checkpoint_test.pth"),
    "after": Path("checkpoints/gain/gain_small_trained.pth"),
}
GAINS = [0.5, 0.75, 1.0, 1.5]

images = sorted(DATA_DIR.glob("kodim*.png"))
assert len(images) == 24, f"预期24张图，实际{len(images)}张"

print("评价设备:", DEVICE)
print("图像数量:", len(images))

# LPIPS 使用 AlexNet 感知距离
lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()

details = []

for name, ckpt_path in CHECKPOINTS.items():
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=True
    )

    model = TrainableGainHyperprior().to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    print(f"\n开始评价: {name}, checkpoint={ckpt_path}")

    for level, gain in enumerate(GAINS):
        for idx, image_path in enumerate(images, 1):
            with Image.open(image_path) as im:
                x = to_tensor(im.convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                encoded = model.compress(x, level)
                decoded = model.decompress(encoded)
                x_hat = decoded["x_hat"]

                assert x_hat.shape == x.shape
                assert torch.isfinite(x_hat).all()

                bits = sum(
                    len(stream) * 8
                    for group in encoded["strings"]
                    for stream in group
                )

                bpp = bits / (x.shape[-2] * x.shape[-1])

                # 统一将重建图像限制在 [0,1] 后计算质量指标
                x_eval = x_hat.clamp(0, 1)
                mse = torch.mean((x - x_eval) ** 2).item()
                psnr = float("inf") if mse == 0 else -10 * math.log10(mse)

                msssim = ms_ssim(
                    x_eval, x, data_range=1.0, size_average=True
                ).item()

                lpips_value = lpips_model(
                    x_eval * 2 - 1, x * 2 - 1
                ).mean().item()

            details.append({
                "model": name,
                "image": image_path.name,
                "level": level,
                "gain": gain,
                "bpp": bpp,
                "psnr_db": psnr,
                "ms_ssim": msssim,
                "lpips": lpips_value,
            })

        print(
            f"{name} | Gain={gain} | "
            f"进度 {idx}/{len(images)}",
            flush=True,
        )

detail_path = OUT_DIR / "gain_ablation_details.csv"
fields = [
    "model", "image", "level", "gain",
    "bpp", "psnr_db", "ms_ssim", "lpips",
]

with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(details)

summary = []
for name in CHECKPOINTS:
    for level, gain in enumerate(GAINS):
        rows = [
            r for r in details
            if r["model"] == name and r["level"] == level
        ]
        summary.append({
            "model": name,
            "level": level,
            "gain": gain,
            "images": len(rows),
            "avg_bpp": sum(r["bpp"] for r in rows) / len(rows),
            "avg_psnr_db": sum(r["psnr_db"] for r in rows) / len(rows),
            "avg_ms_ssim": sum(r["ms_ssim"] for r in rows) / len(rows),
            "avg_lpips": sum(r["lpips"] for r in rows) / len(rows),
        })

summary_path = OUT_DIR / "gain_ablation_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

print("\n===== 训练前后消融汇总 =====")
for r in summary:
    print(
        f"{r['model']:>6} Gain={r['gain']:.2f} "
        f"bpp={r['avg_bpp']:.5f} "
        f"PSNR={r['avg_psnr_db']:.3f} "
        f"MS-SSIM={r['avg_ms_ssim']:.5f} "
        f"LPIPS={r['avg_lpips']:.5f}"
    )

print("\n结果已保存:")
print(detail_path)
print(summary_path)
