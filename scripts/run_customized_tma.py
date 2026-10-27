#!/usr/bin/env python
"""Customized TMA 接入：文本 → 掩膜 → 压缩 → 重建（端到端跑通改进 1 的掩膜接入）。

把改进 1 的文本掩膜生成（CLIPSeg 现在可用，Grounded-SAM2 接口已留）接进
Customized 的 TIC 压缩管线，替换论文原版的 LSeg+TMA，输出重建图 + bpp/PSNR/ROI-PSNR。

依赖（由有显卡的同学运行）：
    torch(CUDA)  compressai  timm  clip  matplotlib  numpy  pillow  transformers
    + third_party/Customizable-ROI-Based-Deep-Image-Compression/（含 model_tic.py / lseg）
    + TIC 预训练权重（如 checkpoints/best_loss_0819_0035.pth.tar，λ=0.0035 → 128 通道）

用法示例：
    # 多类别（点分隔）
    python scripts/run_customized_tma.py \
        --image data/kodak/kodim23.png --prompt "bird.branch" \
        --tic-ckpt checkpoints/best_loss_0819_0035.pth.tar --channels 128

    # 完整句子
    python scripts/run_customized_tma.py \
        --image img.jpg --prompt "the dog on the car" --mode sentence \
        --tic-ckpt checkpoints/best_loss_0819_0035.pth.tar

    # 无显卡兜底（会很慢，仅供冒烟测试）
    ... --device cpu

    # CLIPSeg 走本地权重（避免联网下 HF 模型）
    ... --clipseg-backend local --clipseg-weights /path/to/rd64-uni.pth

阈值语义说明：
    CLIPSeg 输出 sigmoid 概率（背景≈0），默认阈值 0.5；
    论文原版 LSeg 输出余弦相似度（背景基线≈0.8），阈值 0.85。
    本脚本接的是 CLIPSeg，所以默认 --threshold 0.5（与 visualize_mask.py 一致）。

已知坑（跑之前必读，复现阶段发现的第三方代码 bug）：
    model_tic.py:407-408 在 CUDA 上会报 device mismatch：
        similarity_loss = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
    这里的 torch.tensor(1.0/0.01) 没指定 device（默认 CPU），而 similarity 在 CUDA 上，
    torch.where 会抛 "Expected all tensors to be on the same device"。
    修复（一行，与将来 CVA 连续 σ 的改动点相同）：给这两个 tensor 补 device=similarity.device：
        torch.tensor(1.0, device=similarity.device) / torch.tensor(0.01, device=similarity.device)
    （本脚本不改 third_party，保持上游可复现；跑的人本地改这 2 行即可。）
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMIZED = os.path.join(ROOT, "third_party", "Customizable-ROI-Based-Deep-Image-Compression")
for _p in (ROOT, CUSTOMIZED, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.roi.mask_generator import TextMaskGenerator  # noqa: E402
from src.migration.tma_integration import compute_padding, merge_masks, prepare_similarity  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Customized TMA 接入（文本→掩膜→压缩）")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="多类别用点分隔；句子直接写")
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--method", default="clipseg", choices=["clipseg", "grounded_sam2"],
                   help="掩膜生成方法；grounded_sam2 接口已留、实现待补")
    p.add_argument("--clipseg-backend", default="hf", choices=["hf", "local"],
                   help="CLIPSeg 后端：hf=HuggingFace 自动下权重，local=本地权重")
    p.add_argument("--clipseg-weights", default=None, help="local 后端时必填（如 rd64-uni.pth）")

    p.add_argument("--tic-ckpt", required=True, help="TIC 预训练权重 .pth.tar 路径")
    p.add_argument("--channels", type=int, default=128,
                   help="TIC 通道数 N=M。λ=0.0035/0.013 → 128；λ=0.025/0.0483/0.0932 → 192")

    p.add_argument("--sigma", type=float, default=0.01, help="CVA 背景失真权重 σ（ROI 恒为 1.0）")
    p.add_argument("--threshold", type=float, default=0.5, help="掩膜二值化阈值")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")

    p.add_argument("--out", default="out_customized", help="输出目录")
    p.add_argument("--hf-mirror", action="store_true", help="设置 HF_ENDPOINT=hf-mirror（国内下 CLIPSeg）")
    return p.parse_args()


def load_tic(ckpt: str, channels: int, device: str):
    """加载 TIC 并加载 checkpoint（与 eval_kodak.py 一致的兼容加载）。"""
    from model_tic import TIC  # 延迟导入，避免模块级 import clip/lseg 的副作用

    model = TIC(N=channels, M=channels)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt, map_location=device)
    sd = state.get("state_dict", state)

    model_dict = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    n_missing = len(model_dict) - len(compat)
    model_dict.update(compat)
    model.load_state_dict(model_dict)
    print(f"[ckpt] 兼容加载 {len(compat)}/{len(model_dict)} 个键"
          + (f"（{n_missing} 个未匹配，通常是无害的）" if n_missing else ""))

    model.to(device).eval()
    model.update(force=True)  # 更新熵瓶颈（与 eval_kodak.py 一致）
    return model


@torch.no_grad()
def main():
    args = parse_args()
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.startswith("cuda"):
        print("[warn] CUDA 跑之前需先修 model_tic.py:407-408 的 torch.tensor 缺 device 问题（见脚本头部『已知坑』）")

    # ---- 1. 读图 [3,H,W] float [0,1] ----
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = TF.to_tensor(Image.open(args.image).convert("RGB"))
    H, W = img.shape[-2:]

    # ---- 2. 生成文本掩膜（CLIPSeg / Grounded-SAM2 同一入口）----
    prompts = args.prompt.split(".") if args.mode == "multiclass" else [args.prompt]
    gen_kwargs = {"method": args.method, "device": device}
    if args.method == "clipseg":
        gen_kwargs["backend"] = args.clipseg_backend
        gen_kwargs["weights_path"] = args.clipseg_weights
    gen = TextMaskGenerator(**gen_kwargs)
    res = gen.generate(img, prompts, mode=args.mode)
    probs = res.probs.float()                    # [N,H,W]
    merged = merge_masks(probs)                  # [H,W]
    print(f"[mask] {len(prompts)} 个 prompt，合并后覆盖率 {(merged > args.threshold).float().mean():.2%}")

    # ---- 3. pad 图像 + 准备 similarity ----
    pad, out_h, out_w = compute_padding(H, W, min_div=256)
    x_padded = F.pad(img.unsqueeze(0), pad, mode="constant", value=0).to(device)  # [1,3,PH,PW]
    similarity = prepare_similarity(
        merged, pad, sigma=args.sigma, threshold=args.threshold
    ).to(device)                                                                  # [1,1,PH/2,PW/2]

    # ---- 4. 加载模型 + 前向 ----
    model = load_tic(args.tic_ckpt, args.channels, device)
    out = model(x_padded, similarity)
    x_hat = out["x_hat"]  # [1,3,PH,PW]

    # ---- 5. unpad 回原图尺寸 + 算指标 ----
    left, right, top, bottom = pad
    x_hat = F.pad(x_hat, (-left, -right, -top, -bottom))  # [1,3,H,W]
    x_hat = x_hat.clamp(0, 1)

    from src.evaluation.metrics import psnr, roi_psnr

    # 与 eval_kodak.py:242-245 一致：先量化到 8bit 再算 PSNR。
    # metrics.psnr 会对 >1 的 tensor 自动 /255，等价于 paper 的 8bit PSNR 口径。
    x_q = torch.round(img.clamp(0, 1) * 255)
    x_hat_q = torch.round(x_hat.clamp(0, 1) * 255)
    v_psnr = psnr(x_q, x_hat_q)
    v_roi_psnr = roi_psnr(x_q, x_hat_q, (merged > args.threshold))
    num_pixels = H * W
    v_bpp = float(sum(
        (torch.log(l).sum() / (-math.log(2) * num_pixels))
        for l in out["likelihoods"].values()
    ))
    print(f"[result] bpp={v_bpp:.4f}  PSNR={v_psnr:.3f} dB  ROI-PSNR={v_roi_psnr:.3f} dB")

    # ---- 6. 保存重建图 + 掩膜 ----
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    TF.to_pil_image(x_hat[0].cpu()).save(
        os.path.join(args.out, f"{stem}_{args.method}_bpp{v_bpp:.4f}_psnr{v_psnr:.2f}.png"))
    TF.to_pil_image((merged > args.threshold).float().cpu()).save(
        os.path.join(args.out, f"{stem}_{args.method}_mask.png"))
    print(f"[save] -> {args.out}/")


if __name__ == "__main__":
    main()
