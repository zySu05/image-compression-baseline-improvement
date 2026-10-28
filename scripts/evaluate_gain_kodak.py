import csv
from math import log10
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.gain_hyperprior import GainHyperprior


def main():
    data_dir = Path("data/kodak")
    result_dir = Path("results/gain")
    result_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_dir.glob("kodim*.png"))
    assert len(image_paths) == 24, "Kodak 图像数量应为24"

    gain_values = [0.5, 0.75, 1.0, 1.5, 2.0]

    print("加载 Hyperprior Q1 预训练模型")
    codec = GainHyperprior(quality=1, device="cpu")

    summary = []

    for gain in gain_values:
        records = []

        print(f"\n开始评测 gain={gain}")

        for i, path in enumerate(image_paths, start=1):
            with Image.open(path) as im:
                x = to_tensor(im.convert("RGB")).unsqueeze(0)

            encoded = codec.compress(x, gain=gain)
            decoded = codec.decompress(encoded)
            x_hat = decoded["x_hat"].cpu()

            if x_hat.shape != x.shape:
                raise RuntimeError(
                    f"{path.name}: 图像尺寸不匹配"
                )

            mse = torch.mean((x - x_hat) ** 2).item()
            psnr = float("inf") if mse == 0 else 10 * log10(1 / mse)
            bits = codec.count_bits(encoded)
            bpp = bits / (x.shape[-2] * x.shape[-1])

            records.append({
                "image": path.name,
                "gain": gain,
                "bpp": round(bpp, 6),
                "psnr_db": round(psnr, 4)
            })

            print(
                f"{i:02d}/24 {path.name} "
                f"bpp={bpp:.5f} "
                f"PSNR={psnr:.3f} dB",
                flush=True
            )

        csv_path = result_dir / f"gain_{gain:.2f}_kodak.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image", "gain", "bpp", "psnr_db"]
            )
            writer.writeheader()
            writer.writerows(records)

        avg_bpp = sum(r["bpp"] for r in records) / len(records)
        avg_psnr = sum(r["psnr_db"] for r in records) / len(records)

        summary.append({
            "gain": gain,
            "images": len(records),
            "avg_bpp": round(avg_bpp, 6),
            "avg_psnr_db": round(avg_psnr, 4)
        })

        print(
            f"gain={gain:.2f} "
            f"平均bpp={avg_bpp:.5f} "
            f"平均PSNR={avg_psnr:.3f} dB"
        )

    summary_path = result_dir / "gain_kodak_summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== Gain-Hyperprior Kodak完整评测 =====")
    for row in summary:
        print(row)
    print("结果文件：", summary_path)


if __name__ == "__main__":
    main()
