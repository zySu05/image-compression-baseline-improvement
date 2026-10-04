#!/usr/bin/env python
"""统一评测脚本：给定参考图 + 重建图，输出 PSNR / MS-SSIM / LPIPS / ROI-PSNR / bpp。

用法示例：
  python scripts/eval_metrics.py --ref data/kodak/kodim01.png --rec out/recon.png
  python scripts/eval_metrics.py --ref a.png --rec b.png --mask m.png --bytes 12345
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torchvision.transforms.functional as TF

from src.evaluation.metrics import evaluate


def load_img(path: str):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # [3,H,W] float [0,1]


def main():
    p = argparse.ArgumentParser(description="统一评测 PSNR/MS-SSIM/LPIPS/ROI-PSNR/bpp")
    p.add_argument("--ref", required=True, help="参考图路径")
    p.add_argument("--rec", required=True, help="重建图路径")
    p.add_argument("--mask", default=None, help="ROI 掩膜图路径（可选）")
    p.add_argument("--bytes", type=int, default=None, help="码流字节数，用于算 bpp（可选）")
    args = p.parse_args()

    ref = load_img(args.ref)
    rec = load_img(args.rec)
    mask = None
    if args.mask:
        mask = load_img(args.mask)[0:1]  # 取单通道 [1,H,W]

    h, w = ref.shape[-2], ref.shape[-1]
    out = evaluate(ref, rec, mask=mask, num_bytes=args.bytes, size=(h, w))
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k:>10}: {v:.4f}")
        else:
            print(f"{k:>10}: {v}")


if __name__ == "__main__":
    main()
