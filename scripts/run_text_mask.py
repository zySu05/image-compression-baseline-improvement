#!/usr/bin/env python
"""改进 1 入口：文本 → ROI 掩膜（多类别 / 句子输入）。

用法示例：
  # 多类别（点分隔）
  python scripts/run_text_mask.py --image img.jpg --prompt "sky.clouds.hill" --mode multiclass --out out_masks/
  # 完整句子（指代式）
  python scripts/run_text_mask.py --image img.jpg --prompt "the dog on the car" --mode sentence
  # 轻量备选 CLIPSeg
  python scripts/run_text_mask.py --image img.jpg --prompt "tree" --method clipseg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torchvision.transforms.functional as TF

from src.roi.mask_generator import TextMaskGenerator


def main():
    p = argparse.ArgumentParser(description="文本 → ROI 掩膜（改进 1）")
    p.add_argument("--image", required=True, help="输入图像路径")
    p.add_argument("--prompt", required=True, help="多类别用点分隔，如 sky.clouds.hill；句子直接写")
    p.add_argument("--method", default="grounded_sam2", choices=["grounded_sam2", "clipseg"])
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out_masks", help="掩膜输出目录")
    args = p.parse_args()

    img = TF.to_tensor(Image.open(args.image).convert("RGB"))  # [3,H,W]
    prompts = args.prompt.split(".") if args.mode == "multiclass" else [args.prompt]

    gen = TextMaskGenerator(method=args.method, device=args.device)
    res = gen.generate(img, prompts, mode=args.mode)

    os.makedirs(args.out, exist_ok=True)
    for i, (m, lab) in enumerate(zip(res.masks, res.labels)):
        TF.to_pil_image(m).save(os.path.join(args.out, f"{i:02d}_{lab.replace(' ', '_')}.png"))
    print(f"生成 {len(res.masks)} 个掩膜到 {args.out}/")


if __name__ == "__main__":
    main()
