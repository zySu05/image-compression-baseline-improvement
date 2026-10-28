import csv
from math import log10
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.continuous_gain import (
    ContinuousGainHyperprior,
)

checkpoint = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True
)

model = ContinuousGainHyperprior().eval()
model.load_state_dict(checkpoint["model"])

with Image.open("data/kodak/kodim01.png") as im:
    x = to_tensor(im.convert("RGB")).unsqueeze(0)

positions = [0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]

rows = []

for position in positions:
    encoded = model.compress_at(x, position)
    decoded = model.decompress_at(encoded)
    x_hat = decoded["x_hat"]

    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()

    bits = sum(
        len(stream) * 8
        for group in encoded["strings"]
        for stream in group
    )

    bpp = bits / (x.shape[-2] * x.shape[-1])
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = float("inf") if mse == 0 else 10 * log10(1 / mse)

    rows.append({
        "position": position,
        "bpp": bpp,
        "psnr_db": psnr,
    })

    print(
        f"Position={position:.2f} | "
        f"bpp={bpp:.5f} | "
        f"PSNR={psnr:.3f} dB",
        flush=True
    )

out = Path("results/training/continuous_gain_kodim01.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["position", "bpp", "psnr_db"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("连续码率实验结果已保存:", out)
