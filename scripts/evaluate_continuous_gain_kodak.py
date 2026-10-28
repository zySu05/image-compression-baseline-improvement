import csv
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.continuous_gain import ContinuousGainHyperprior

torch.set_num_threads(4)

POSITIONS = [0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
DATA_DIR = Path("data/kodak")
OUT_DIR = Path("results/continuous")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True,
)

model = ContinuousGainHyperprior()
model.load_state_dict(ckpt["model"])
model.eval()

images = sorted(DATA_DIR.glob("kodim*.png"))
assert len(images) == 24, f"应有24张图像，实际为{len(images)}张"

details = []

with torch.no_grad():
    for idx, path in enumerate(images, 1):
        with Image.open(path) as im:
            x = to_tensor(im.convert("RGB")).unsqueeze(0)

        for position in POSITIONS:
            encoded = model.compress_at(x, position)
            x_hat = model.decompress_at(encoded)["x_hat"]

            assert x_hat.shape == x.shape

            bits = sum(
                len(s) * 8
                for group in encoded["strings"]
                for s in group
            )

            bpp = bits / (x.shape[-2] * x.shape[-1])
            mse = torch.mean((x - x_hat) ** 2).item()
            psnr = (
                float("inf") if mse == 0
                else -10 * math.log10(mse)
            )

            details.append({
                "image": path.name,
                "position": position,
                "bpp": bpp,
                "psnr_db": psnr,
            })

        print(f"[{idx:02d}/24] {path.name} 已完成", flush=True)

detail_path = OUT_DIR / "continuous_kodak_details.csv"
with detail_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "position", "bpp", "psnr_db"]
    )
    writer.writeheader()
    writer.writerows(details)

summary = []
print("\nPosition   平均bpp   平均PSNR")
for position in POSITIONS:
    rows = [
        r for r in details
        if r["position"] == position
    ]

    avg_bpp = sum(r["bpp"] for r in rows) / len(rows)
    avg_psnr = sum(r["psnr_db"] for r in rows) / len(rows)

    summary.append({
        "position": position,
        "images": len(rows),
        "avg_bpp": avg_bpp,
        "avg_psnr_db": avg_psnr,
    })

    print(f"{position:7.2f}   {avg_bpp:.6f}   {avg_psnr:.4f}")

summary_path = OUT_DIR / "continuous_kodak_summary.csv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "position", "images", "avg_bpp", "avg_psnr_db"
        ]
    )
    writer.writeheader()
    writer.writerows(summary)

print("\n保存完成:")
print(detail_path)
print(summary_path)
