import csv
from math import log10
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.gain_hyperprior import GainHyperprior


def main():
    image_path = Path("data/kodak/kodim01.png")
    output_dir = Path("results/gain")
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        x = to_tensor(im.convert("RGB")).unsqueeze(0)

    print("正在加载同一套 Hyperprior Q1 权重...")
    codec = GainHyperprior(quality=1, device="cpu")

    rows = []

    for gain in [0.5, 0.75, 1.0, 1.5, 2.0]:
        encoded = codec.compress(x, gain=gain)
        decoded = codec.decompress(encoded)
        x_hat = decoded["x_hat"].cpu()

        if x_hat.shape != x.shape:
            raise RuntimeError("重建图像尺寸不匹配")

        mse = torch.mean((x - x_hat) ** 2).item()
        psnr = float("inf") if mse == 0 else 10 * log10(1 / mse)
        bits = codec.count_bits(encoded)
        bpp = bits / (x.shape[-2] * x.shape[-1])

        rows.append({
            "gain": gain,
            "bpp": bpp,
            "psnr_db": psnr,
        })

        print(
            f"gain={gain:.2f} | "
            f"bpp={bpp:.5f} | "
            f"PSNR={psnr:.3f} dB",
            flush=True
        )

    csv_path = output_dir / "gain_kodim01.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gain", "bpp", "psnr_db"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("实验结果已保存：", csv_path)


if __name__ == "__main__":
    main()
