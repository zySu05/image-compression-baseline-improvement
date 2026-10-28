#!/usr/bin/env python
"""LMA 小规模训练：只训练 simi_net + sub_impor_net，冻结 base 和 gain。

对应原论文 train_tic.py 的 ROI 加权损失：
    loss = bpp + lambda * 255^2 * mean( (x_hat - x)^2 * weight )
    weight = ROI 掩膜（ROI=1，背景=loss_sigma，默认 0.01，背景失真基本不惩罚）

训练的是 src/compression/lma_adapter.py 里 LMAAdapter 的两个新增模块：
  - simi_net      ：文本掩膜 -> 隐空间相似度图
  - sub_impor_net ：潜变量 -> 内容重要性图
base（bmshj2018 主干）和 log_gain（可变码率 gain）都冻结不动。

用法（在配好 torch/compressai/transformers + CLIPSeg 的机器上，项目根目录）：
    python scripts/train_lma.py \
        --images-dir data/train/div2k \
        --prompt "a building" \
        --gain-ckpt checkpoints/gain/gain_small_trained.pth \
        --steps 200 --out checkpoints/lma

训练完把 checkpoints/lma/lma_trained.pth 回传，用 run_tma_lma.py 的 --gain-ckpt 加载即可。
"""
import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor

from src.compression.lma_adapter import LMAAdapter
from src.roi.mask_generator import TextMaskGenerator
from src.migration.tma_integration import prepare_cva_mask


def roi_rate_distortion_loss(output, target, mask, lmbda, loss_sigma=0.01):
    """ROI 加权率失真损失（对应原论文 train_tic.py 的 RateDistortionLoss）。"""
    N, _, H, W = target.shape
    pixels = N * H * W

    bpp = -sum(
        torch.log2(lik.clamp_min(1e-9)).sum()
        for lik in output["likelihoods"].values()
    ) / pixels

    # ROI=1，背景=loss_sigma（默认 0.01，背景失真近乎不惩罚）
    weight = torch.where(
        mask > 0.5,
        torch.tensor(1.0, device=mask.device, dtype=mask.dtype),
        torch.tensor(loss_sigma, device=mask.device, dtype=mask.dtype),
    )
    mse = torch.mean(((output["x_hat"] - target) * weight) ** 2)

    loss = bpp + lmbda * 255 ** 2 * mse
    return loss, bpp, mse


def parse_args():
    p = argparse.ArgumentParser(description="LMA 小规模训练（只训 simi_net + sub_impor_net）")
    p.add_argument("--images-dir", default="data/train/div2k", help="训练图像目录")
    p.add_argument("--num-images", type=int, default=16, help="最多用多少张图")
    p.add_argument("--prompt", default="a building", help="生成 ROI 掩膜的文本提示")
    p.add_argument("--crop-size", type=int, default=256, help="训练裁剪尺寸（需能被 16 整除）")

    p.add_argument("--gain-ckpt", default="checkpoints/gain/gain_small_trained.pth",
                   help="Gain checkpoint（含 base + log_gain，LMA 模块随机初始化）")
    p.add_argument("--steps", type=int, default=200, help="训练步数")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率")
    p.add_argument("--lmbda", type=float, default=0.0035, help="率失真 lambda")
    p.add_argument("--loss-sigma", type=float, default=0.01,
                   help="损失里背景权重（ROI=1，背景=该值）")
    p.add_argument("--position", type=float, default=1.5, help="训练时的全局 gain 档位")

    p.add_argument("--clipseg-backend", default="hf", choices=["hf", "local"])
    p.add_argument("--clipseg-weights", default=None, help="local 后端时必填")
    p.add_argument("--hf-mirror", action="store_true", help="国内下 CLIPSeg 用镜像")

    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--out", default="checkpoints/lma", help="checkpoint 输出目录")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_mirror:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted(Path(args.images_dir).rglob("*.png"))[: args.num_images]
    if len(images) == 0:
        raise RuntimeError(f"训练图像为空：{args.images_dir}")
    print(f"[data] {len(images)} 张训练图，设备 {device}", flush=True)

    # ---- 模型：加载 gain checkpoint，冻结 base + gain，只留 simi/sub 可训 ----
    try:
        state = torch.load(args.gain_ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.gain_ckpt, map_location=device)
    model = LMAAdapter()
    missing, _ = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[model] LMA 新增模块随机初始化：{len(missing)} 个参数（simi_net/sub_impor_net）")

    model.log_gain.requires_grad_(False)  # 冻结 gain
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] 可训练参数数：{sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    model = model.to(device).train()

    # ---- 掩膜生成器（CLIPSeg，推理用，不参与梯度）----
    gen = TextMaskGenerator(
        method="clipseg",
        device=device,
        backend=args.clipseg_backend,
        weights_path=args.clipseg_weights,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    print("[train] 开始训练", flush=True)

    for step in range(1, args.steps + 1):
        path = random.choice(images)
        with Image.open(path) as im:
            crop = RandomCrop(args.crop_size)(im.convert("RGB"))
        img = to_tensor(crop)  # [3,H,W] float [0,1]，先留在 CPU 供 CLIPSeg

        with torch.no_grad():
            res = gen.generate(img, [args.prompt], mode="multiclass")
            mask = prepare_cva_mask(res.probs.float(), threshold=0.5)  # [1,1,H,W] 0/1

        x = img.unsqueeze(0).to(device)        # [1,3,H,W]
        mask = mask.to(device)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")

        optimizer.zero_grad(set_to_none=True)
        output = model(x, mask, args.position)   # 前向，梯度流向 simi/sub
        loss, bpp, mse = roi_rate_distortion_loss(
            output, x, mask, args.lmbda, args.loss_sigma
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        history.append({"step": step, "loss": loss.item(), "bpp": bpp.item(), "mse": mse.item()})
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"Step {step:03d}/{args.steps} | loss={loss.item():.5f} "
                  f"bpp={bpp.item():.5f} mse={mse.item():.6f}", flush=True)

    # ---- 保存 ----
    ckpt = out_dir / "lma_trained.pth"
    torch.save({
        "step": args.steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lmbda": args.lmbda,
        "loss_sigma": args.loss_sigma,
        "position": args.position,
    }, ckpt)

    csv_path = out_dir / "lma_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "bpp", "mse"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\n[done] checkpoint: {ckpt}")
    print(f"[done] 训练记录: {csv_path}")
    print("[note] 用 run_tma_lma.py --gain-ckpt 指向上面这个 checkpoint 即可加载训练好的 LMA")


if __name__ == "__main__":
    main()
