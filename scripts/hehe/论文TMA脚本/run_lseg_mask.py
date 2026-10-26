#!/usr/bin/env python
"""测试论文 Customized 的 TMA 模块：LSeg + CLIP 文本定位。

复现 eval_kodak.py:184-195 的掩膜生成逻辑：
    img_feat  = LSegNet(x_padded)                      # [1,512,Hf,Wf]
    text_feat = clip_pretrained.encode_text(tokenize(labels))
    sim       = cosine(img_feat, text_feat)            # [N,H,W]，范围 [-1,1]
    mask      = sim > 0.85                             # 论文 CVA 的阈值

输出与 CLIPSeg 版本同格式（MaskResult），可直接拼在同一张对比图里。

用法：
  python scripts/run_lseg_mask.py --image data/kodak/kodim23.png --prompt "bird.branch"
  python scripts/run_lseg_mask.py --image img.jpg --prompt "the dog on the car" --mode sentence
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSEG_REPO = os.path.join(ROOT, "third_party", "Customizable-ROI-Based-Deep-Image-Compression")
sys.path.insert(0, ROOT)
sys.path.insert(0, LSEG_REPO)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_CKPT = os.path.join(LSEG_REPO, "checkpoints", "lseg_minimal_e200.ckpt")
PAD_DIV = 256          # 与 eval_kodak.py 的 compute_padding(min_div=256) 保持一致


def parse_args():
    p = argparse.ArgumentParser(description="LSeg + CLIP 文本定位（论文 TMA）")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="多类别用点分隔；句子直接写")
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="lseg_minimal_e200.ckpt 路径")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.85, help="论文 CVA 的余弦阈值")
    p.add_argument("--out", default="out_masks")
    p.add_argument("--size", type=int, default=365)
    p.add_argument("--save-npy", action="store_true", help="导出原始余弦相似度 .npy")
    p.add_argument(
        "--absolute",
        action="store_true",
        help="按原始余弦值(0~1)上色。默认按每类的[min,max]归一化，"
        "因为 LSeg 余弦有 ~0.8 的高基线，绝对上色会整张图全红",
    )
    p.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="允许 timm 下载 ImageNet ViT-L 权重（1.2GB）。默认关闭，因为 ckpt 里已有全量权重",
    )
    return p.parse_args()


def build_model(ckpt: str, device: str, allow_pretrained: bool):
    """构建 LSegNet 并加载论文 checkpoint。"""
    import timm

    if not allow_pretrained:
        # 避免下载 1.2GB 的 ImageNet ViT-L 权重：ckpt 会覆盖它
        _orig_create = timm.create_model

        def _create_no_pretrain(*args, **kwargs):
            kwargs["pretrained"] = False
            return _orig_create(*args, **kwargs)

        timm.create_model = _create_no_pretrain

    from lseg.lseg_net import LSegNet   # noqa: E402  (依赖上面的 timm patch)

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=256,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"找不到 LSeg checkpoint: {ckpt}\n"
            "下载：https://hf-mirror.com/IsaacLabe/Lseg_minimal_model/resolve/main/lseg_minimal_e200.ckpt"
        )

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    try:
        net.load_state_dict(state, strict=True)
        print("[ckpt] strict=True 加载成功（图像骨干权重全部来自 ckpt）")
    except RuntimeError as e:
        print(f"[ckpt] strict 失败，改用 strict=False：{str(e)[:160]}")
        net.load_state_dict(state, strict=False)

    net = net.eval().to(device)
    # clip.load(device="cuda") 返回的是 fp16 模型（clip 只在 cpu 上才 .float()），
    # 而 LSeg 的图像特征是 fp32，直接算余弦会报 dtype 不匹配。
    net.clip_pretrained.float()
    return net


def compute_padding(h: int, w: int, min_div: int = 256):
    """与 eval_kodak.py:39 完全一致：对称 padding 到 min_div 的整数倍。"""
    out_h = (h + min_div - 1) // min_div * min_div
    out_w = (w + min_div - 1) // min_div * min_div
    left, top = (out_w - w) // 2, (out_h - h) // 2
    right, bottom = out_w - w - left, out_h - h - top
    return (left, right, top, bottom), out_h, out_w


@torch.no_grad()
def lseg_similarity(net, image: torch.Tensor, labels, device: str):
    """image: [3,H,W] float 0~1 -> (sim [N,H,W] 余弦相似度, 特征图尺寸)。

    严格按 eval_kodak.py:179-195：先在对称 padding 后的图上算 similarity，
    再把它裁剪回原图坐标（论文里是让 TIC 模型内部处理，最后 unpad）。
    """
    import clip

    h, w = image.shape[-2:]
    pad, out_h, out_w = compute_padding(h, w, PAD_DIV)
    left, _, top, _ = pad
    x = F.pad(image, pad, mode="constant", value=0).unsqueeze(0).to(device)

    img_feat = F.normalize(net(x), dim=1)                                  # [1,512,Hf,Wf]
    tokens = clip.tokenize(labels).to(device)
    text_feat = F.normalize(net.clip_pretrained.encode_text(tokens), dim=1)  # [N,512]

    sim = torch.einsum("nc,nchw->nhw", text_feat, img_feat)                # [N,Hf,Wf]
    sim = F.interpolate(
        sim.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=False
    ).squeeze(1)
    sim = sim[:, top:top + h, left:left + w]
    return sim, tuple(img_feat.shape)


def main():
    args = parse_args()

    import torchvision.transforms.functional as TF
    from PIL import Image

    from src.roi.mask_generator import MaskResult
    from visualize_mask import heatmap, panel

    image = TF.to_tensor(Image.open(args.image).convert("RGB"))            # [3,H,W]
    prompts = args.prompt.split(".") if args.mode == "multiclass" else [args.prompt]

    net = build_model(args.ckpt, args.device, args.pretrained_backbone)
    sim, feat_shape = lseg_similarity(net, image, prompts, args.device)     # [N,H,W]
    sim_cpu = sim.float().cpu()
    print(f"[feature] LSeg 图像特征 {feat_shape}  <- 原图 {tuple(image.shape)}；"
          f"掩膜由这张低分辨率图插值得到")

    masks = (sim_cpu > args.threshold).float()
    scores = sim_cpu.flatten(1).max(1).values.tolist()
    res = MaskResult(
        masks=masks,
        labels=list(prompts),
        scores=scores,
        probs=sim_cpu.clamp(0, 1),   # 仅为可视化；原始值在 sim_cpu 里
    )

    print(f"\n{'label':22s} {'coverage':>9s} {'sim_max':>8s} {'sim_p50':>8s} {'sim_mean':>9s}")
    for lab, m, s in zip(res.labels, res.masks, sim_cpu):
        print(
            f"{lab:22s} {m.mean():9.2%} {s.max():8.3f} "
            f"{s.median():8.3f} {s.mean():9.3f}"
        )
    print(f"\n阈值 = {args.threshold}（论文 CVA 用的余弦阈值）")
    print("注意：sim_p50（中位数）就是全图的背景基线，LSeg 余弦通常在 0.8 以上，")
    print("      所以真正有判别力的是 sim_max - sim_p50 的差，不是绝对值。")

    def to_display(s):
        if args.absolute:
            return s.clamp(0, 1)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo).clamp_min(1e-6)

    size = (args.size, args.size)
    # 第一行：相对显著性热力图；第二行：阈值二值掩膜
    row1 = [panel(image, "input", size)]
    row2 = [panel(torch.zeros_like(image), f"mask @ {args.threshold}", size)]
    for lab, s, m in zip(res.labels, sim_cpu, res.masks):
        row1.append(panel(heatmap(image, to_display(s)), f"{lab} ({s.max():.2f})", size))
        row2.append(panel(m.unsqueeze(0).repeat(3, 1, 1), f"{lab} mask ({m.mean():.1%})", size))
    merged = sim_cpu.max(dim=0).values
    merged_mask = (merged > args.threshold).float()
    row1.append(panel(heatmap(image, to_display(merged)), f"merged ({merged.max():.2f})", size))
    row2.append(
        panel(merged_mask.unsqueeze(0).repeat(3, 1, 1),
              f"merged mask ({merged_mask.mean():.1%})", size)
    )

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    ncol, cell_h = len(row1), size[1] + 20
    sheet = Image.new("RGB", (size[0] * ncol, cell_h * 2), "white")
    for i, im in enumerate(row1):
        sheet.paste(im, (i * size[0], 0))
    for i, im in enumerate(row2):
        sheet.paste(im, (i * size[0], cell_h))
    out_png = os.path.join(args.out, f"{stem}_lseg_grid.png")
    sheet.save(out_png)
    print(f"grid -> {out_png}")

    if args.save_npy:
        import numpy as np

        out_npy = os.path.join(args.out, f"{stem}_lseg_sim.npy")
        np.save(out_npy, sim_cpu.numpy())
        print(f"sim  -> {out_npy}  shape={tuple(sim_cpu.shape)}")


if __name__ == "__main__":
    main()
