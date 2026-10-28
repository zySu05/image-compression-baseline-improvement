import argparse
import csv
from math import log10
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.trainable_gain_hyperprior import TrainableGainHyperprior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    device = "cpu"
    path = Path("checkpoints/gain/gain_small_trained.pth")
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )

    model = TrainableGainHyperprior().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_paths = sorted(Path("data/kodak").glob("kodim*.png"))
    assert len(image_paths) == 24
    assert 1 <= args.limit <= 24
    image_paths = image_paths[:args.limit]

    results_dir = Path("results/training")
    results_dir.mkdir(parents=True, exist_ok=True)

    gains = [0.5, 0.75, 1.0, 1.5]
    summary = []
    details = []

    print("加载训练后的模型成功")
    print("训练步数:", checkpoint["step"])

    for level in range(4):
        records = []

        for i, image_path in enumerate(image_paths, 1):
            with Image.open(image_path) as im:
                x = to_tensor(im.convert("RGB")).unsqueeze(0)

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
                mse = torch.mean((x - x_hat) ** 2).item()
                psnr = float("inf") if mse == 0 else 10 * log10(1 / mse)

            records.append((bpp, psnr))
            details.append({
                "image": image_path.name,
                "level": level,
                "bpp": bpp,
                "psnr_db": psnr
            })

            print(
                f"Level={level} {i:02d}/{len(image_paths)} "
                f"{image_path.name} "
                f"bpp={bpp:.5f} PSNR={psnr:.3f}",
                flush=True
            )

        avg_bpp = sum(r[0] for r in records) / len(records)
        avg_psnr = sum(r[1] for r in records) / len(records)

        summary.append({
            "level": level,
            "initial_gain": gains[level],
            "images": len(records),
            "avg_bpp": avg_bpp,
            "avg_psnr_db": avg_psnr
        })

    for name, rows in [
        ("trained_gain_summary.csv", summary),
        ("trained_gain_details.csv", details)
    ]:
        output = results_dir / name
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("保存:", output)

    print("\n===== 训练后真实码率评测 =====")
    for r in summary:
        print(
            f"Level={r['level']} "
            f"bpp={r['avg_bpp']:.5f} "
            f"PSNR={r['avg_psnr_db']:.3f} dB"
        )


if __name__ == "__main__":
    main()
