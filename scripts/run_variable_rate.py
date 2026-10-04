#!/usr/bin/env python
"""改进 3 入口：CompressAI 多 quality 跑 RD 曲线（可变码率基准）。

用法示例：
  python scripts/run_variable_rate.py --arch cheng2020-anchor --qualities 1,3,5,6 --images "data/kodak/*.png"
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torchvision.transforms.functional as TF

from src.compression.variable_rate import load_model, compress_one, bitrate


def _psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(10 * (1.0 / mse).log10()) if mse.item() > 0 else float("inf")


def main():
    p = argparse.ArgumentParser(description="可变码率 RD 曲线（改进 3）")
    p.add_argument("--arch", default="cheng2020-anchor",
                   help="CompressAI 底座，如 cheng2020-anchor / bmshj2018-factorized")
    p.add_argument("--qualities", default="1,3,5,6", help="逗号分隔的 quality 列表")
    p.add_argument("--images", required=True, help="图像 glob，如 data/kodak/*.png")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    files = sorted(glob.glob(args.images))
    if not files:
        raise SystemExit(f"未匹配到图像：{args.images}")
    qualities = [int(q) for q in args.qualities.split(",")]

    print(f"{'quality':>8} {'bpp':>8} {'psnr':>8}")
    for q in qualities:
        model = load_model(args.arch, q, args.device)
        bpps, psnrs = [], []
        for f in files:
            x = TF.to_tensor(Image.open(f).convert("RGB")).unsqueeze(0).to(args.device)
            x_hat, strings = compress_one(model, x)
            num_pixels = x.shape[-2] * x.shape[-1]
            bpps.append(bitrate(strings, num_pixels))
            psnrs.append(_psnr(x.cpu(), x_hat.cpu()))
        print(f"{q:>8} {sum(bpps) / len(bpps):>8.4f} {sum(psnrs) / len(psnrs):>8.4f}")


if __name__ == "__main__":
    main()
