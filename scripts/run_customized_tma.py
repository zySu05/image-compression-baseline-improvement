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
        --tic-ckpt checkpoints/best_loss_0819_0035.pth.tar
    # （N/M 默认 128/192，与 best_loss_0819_0035 一致，无需手动指定）

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
    p.add_argument("--channels-n", type=int, default=128,
                   help="TIC 潜变量通道 N（best_loss_0819_0035 为 128）")
    p.add_argument("--channels-m", type=int, default=192,
                   help="TIC 扩展通道 M（论文各 λ 均为 192）")

    p.add_argument("--sigma", type=float, default=0.01, help="CVA 背景失真权重 σ（ROI 恒为 1.0）")
    p.add_argument("--threshold", type=float, default=0.5, help="掩膜二值化阈值")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")

    p.add_argument("--out", default="out_customized", help="输出目录")
    p.add_argument("--hf-mirror", action="store_true", help="设置 HF_ENDPOINT=hf-mirror（国内下 CLIPSeg）")
    return p.parse_args()


def load_tic(ckpt: str, channels_n: int, channels_m: int, device: str):
    """加载 TIC 并加载 checkpoint（与 eval_kodak.py 一致的兼容加载）。"""
    from model_tic import TIC  # 延迟导入，避免模块级 import clip/lseg 的副作用

    model = TIC(N=channels_n, M=channels_m)  # N 与 M 不一定相等，以 checkpoint 形状为准
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt, map_location=device)
    sd = state.get("state_dict", state)

    # compressai 1.2.0b3 -> 1.2.8：熵模型参数改名。
    # 不改的话这 14 个键名字对不上，会被下面的 shape 过滤静默丢弃，
    # 熵模型用随机参数算 likelihoods，bpp 全错（PSNR 却正常，问题很隐蔽）。
    renamed = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("entropy_bottleneck."):
            for old, new in (("_matrix", "matrices."), ("_bias", "biases."), ("_factor", "factors.")):
                if k.startswith(f"entropy_bottleneck.{old}"):
                    nk = f"entropy_bottleneck.{new}{k[len(f'entropy_bottleneck.{old}'):]}"
                    break
        renamed[nk] = v
    sd = renamed

    model_dict = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compat)
    model.load_state_dict(model_dict)

    # 诊断：区分「本来就不用」和「真的没加载上」
    from collections import Counter
    unmatched = [k for k in sd if k not in compat]        # ckpt 里有、模型没接收
    missing = [k for k in model_dict if k not in compat]  # 模型有、ckpt 没覆盖
    print(f"[ckpt] 匹配 {len(compat)}/{len(model_dict)} 个键")
    if unmatched:
        print(f"[ckpt] ckpt 未被模型接收: {dict(Counter(k.split('.')[0] for k in unmatched))}")
        print(f"[ckpt]   前 5 个: {unmatched[:5]}")
    if missing:
        non_lseg = [k for k in missing if not k.startswith("net_lseg.")]
        if non_lseg:
            print(f"[ckpt] 模型未被 ckpt 覆盖 {len(missing)} 个键，其中非 net_lseg: "
                  f"{dict(Counter(k.split('.')[0] for k in non_lseg))} ← 检查 N/M 与键名")
        else:
            print(f"[ckpt] 模型未被 ckpt 覆盖 {len(missing)} 个键（全是 net_lseg，推理不用，无害）")

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
    model = load_tic(args.tic_ckpt, args.channels_n, args.channels_m, device)
    out = model(x_padded, similarity)
    x_hat = out["x_hat"]  # [1,3,PH,PW]

    # ---- 5. unpad 回原图尺寸 + 算指标 ----
    left, right, top, bottom = pad
    x_hat = F.pad(x_hat, (-left, -right, -top, -bottom))  # [1,3,H,W]
    x_hat = x_hat.clamp(0, 1).cpu()  # 指标在 CPU 算，避免与 CPU 上的 x_q/mask 混算报错

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
