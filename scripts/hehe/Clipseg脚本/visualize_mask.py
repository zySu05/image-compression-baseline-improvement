#!/usr/bin/env python
"""改进 1 可视化：原图 + 每类热力图 + 阈值二值掩膜。

输出格式与 scripts/run_lseg_mask.py 完全对齐（两行 × (1+N+1) 列 + 同一张统计表），
便于把 CLIPSeg 和论文的 LSeg+TMA 放在同一口径下比较：

    行 1：input | 类别 1..N 的相对显著性 | merged
    行 2：阈值 | 类别 1..N 的二值掩膜     | merged

统计表四列：coverage（阈值以上像素占比）、score_max、score_p50、score_mean。

注意两个模型的分数语义不同：
    CLIPSeg  score = sigmoid 概率，背景接近 0，默认阈值 0.5
    LSeg     score = CLIP 余弦相似度，背景基线约 0.8，论文阈值 0.85
所以「阈值」本身不可跨模型直接比，能比的是覆盖率、分布形状和掩膜位置。

用法：
  python scripts/visualize_mask.py --image img.jpg --prompt "bowl.jar.fork"
  python scripts/visualize_mask.py --image img.jpg --prompt "the dog on the car" --mode sentence
  python scripts/visualize_mask.py --image img.jpg --prompt "bowl.jar" --threshold 0.5 --save-npy
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from src.roi.mask_generator import TextMaskGenerator


def heatmap(base: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """base [3,H,W] 原图；prob [H,W] 0~1 软掩膜 -> 红色高亮叠加。"""
    p = prob.unsqueeze(0)
    color = torch.tensor([1.0, 0.12, 0.12], dtype=base.dtype, device=base.device).view(3, 1, 1)
    return (base * (1 - p * 0.75) + color * (p * 0.75)).clamp(0, 1)


def panel(t: torch.Tensor, label: str, size) -> Image.Image:
    img = TF.to_pil_image(t.clamp(0, 1).cpu()).resize(size, Image.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 20), "white")
    canvas.paste(img, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def to_display(prob: torch.Tensor, normalize: bool) -> torch.Tensor:
    """归一到 0~1 用于上色。

    默认按绝对概率（sigmoid 输出本身就有意义）；
    --normalize 改成每类 min-max，便于和 LSeg 的相对显著性图对照。
    """
    if normalize:
        lo, hi = prob.min(), prob.max()
        return (prob - lo) / (hi - lo).clamp_min(1e-6)
    return prob.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description="文本 → ROI 掩膜可视化")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="多类别用点分隔；句子直接写")
    p.add_argument("--method", default="clipseg", choices=["grounded_sam2", "clipseg"])
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--threshold", type=float, default=0.5,
                   help="二值化阈值；CLIPSeg 是概率，默认 0.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out_masks")
    p.add_argument("--size", type=int, default=365, help="每个面板的边长")
    p.add_argument("--normalize", action="store_true",
                   help="热力图按每类 min-max 归一化（默认按绝对概率上色）")
    p.add_argument("--save-npy", action="store_true", help="导出 probs.npy 供 TMA 使用")
    args = p.parse_args()

    img = TF.to_tensor(Image.open(args.image).convert("RGB"))  # [3,H,W] 0~1
    prompts = args.prompt.split(".") if args.mode == "multiclass" else [args.prompt]

    gen = TextMaskGenerator(method=args.method, device=args.device)
    res = gen.generate(img, prompts, mode=args.mode)
    probs = res.probs.float().cpu()                     # [N,H,W]

    # ---- 统计表（与 run_lseg_mask.py 同口径）----
    print(f"\n{'label':22s} {'coverage':>9s} {'score_max':>9s} "
          f"{'score_p50':>9s} {'score_mean':>10s}")
    masks = {}
    for lab, pr in zip(res.labels, probs):
        m = (pr > args.threshold)
        masks[lab] = m
        print(f"{lab:22s} {m.float().mean():9.2%} {pr.max():9.3f} "
              f"{pr.median():9.3f} {pr.mean():10.3f}")
    print(f"\n阈值 = {args.threshold}")
    if args.method == "clipseg":
        print("score 是 sigmoid 概率（背景接近 0）；对比 LSeg 时注意它输出的是余弦，")
        print("两者阈值不可直接互换，可比的是覆盖率、score_max - score_p50 与掩膜位置。")

    size = (args.size, args.size)
    merged = probs.max(dim=0).values
    merged_mask = (merged > args.threshold)

    row1 = [panel(img, "input", size)]
    row2 = [panel(torch.zeros_like(img), f"mask @ {args.threshold}", size)]
    for lab, pr in zip(res.labels, probs):
        row1.append(panel(heatmap(img, to_display(pr, args.normalize)),
                          f"{lab} ({pr.max():.2f})", size))
        m = masks[lab].float()
        row2.append(panel(m.unsqueeze(0).repeat(3, 1, 1),
                          f"{lab} mask ({m.mean():.1%})", size))
    row1.append(panel(heatmap(img, to_display(merged, args.normalize)),
                      f"merged ({merged.max():.2f})", size))
    row2.append(panel(merged_mask.float().unsqueeze(0).repeat(3, 1, 1),
                      f"merged mask ({merged_mask.float().mean():.1%})", size))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    ncol, cell_h = len(row1), size[1] + 20
    sheet = Image.new("RGB", (size[0] * ncol, cell_h * 2), "white")
    for i, im in enumerate(row1):
        sheet.paste(im, (i * size[0], 0))
    for i, im in enumerate(row2):
        sheet.paste(im, (i * size[0], cell_h))
    out_png = os.path.join(args.out, f"{stem}_grid.png")
    sheet.save(out_png)
    print(f"grid -> {out_png}")

    if args.save_npy:
        import numpy as np

        np.save(os.path.join(args.out, f"{stem}_probs.npy"), probs.numpy())
        print(f"probs -> {args.out}/{stem}_probs.npy  shape={tuple(probs.shape)}")


if __name__ == "__main__":
    main()
