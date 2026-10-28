#!/usr/bin/env python
"""TMA → LMA 对接：文本 → 掩膜（改进1）→ 隐空间掩膜注意力压缩（LMA 迁移）。

把吴多迅的 TextMaskGenerator（TMA，CLIPSeg）接到 LMAAdapter（隐空间掩膜注意力），
替换 scripts/run_joint_pipeline.py 里内联的 CLIPSeg，统一走 src/roi/mask_generator.py。

链路：
    文本 prompt → TextMaskGenerator → MaskResult.probs [N,H,W]
               → prepare_cva_mask → mask [1,1,H,W]
               → LMAAdapter.compress_lma(x, mask, position) → 编码
               → decompress_lma → x_hat → bpp/PSNR/ROI-PSNR

与 run_tma_cva.py（CVA σ）的区别：
  - 这里走 LMA（隐空间乘法调制 y * (imp + sim)），解码端不需要掩膜、不占额外码流；
  - 没有 σ 参数（σ 是 CVA 的空间增益概念，LMA 由 simi_net 隐式完成 ROI 加权）。

注意：
  - LMAAdapter 新增的 simi_net / sub_impor_net 不在现有 gain checkpoint 里，随机初始化，
    本脚本只验证「结构跑通」，性能需小规模训练后才能评估。

用法：
    python scripts/run_tma_lma.py --image data/kodak/kodim01.png --prompt "a building" \
        --position 1.5 --out results/tma_lma
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.roi.mask_generator import TextMaskGenerator              # noqa: E402
from src.migration.tma_integration import prepare_cva_mask        # noqa: E402
from src.compression.lma_adapter import LMAAdapter                # noqa: E402
from src.evaluation.metrics import psnr, roi_psnr                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TMA→LMA 对接（文本→掩膜→隐空间调制压缩）")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="多类别用点分隔；句子直接写")
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--method", default="clipseg", choices=["clipseg", "grounded_sam2"],
                   help="掩膜生成方法；grounded_sam2 接口已留、实现待补")
    p.add_argument("--clipseg-backend", default="hf", choices=["hf", "local"],
                   help="CLIPSeg 后端：hf=HuggingFace 自动下权重，local=本地权重")
    p.add_argument("--clipseg-weights", default=None, help="local 后端时必填")

    p.add_argument("--gain-ckpt", default="checkpoints/gain/gain_small_trained.pth",
                   help="Gain checkpoint（含 base + log_gain，不含 LMA 新增模块）")
    p.add_argument("--position", type=float, default=1.5, help="全局 Gain 档位 0~3")
    p.add_argument("--threshold", type=float, default=0.5, help="掩膜二值化阈值")

    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--out", default="results/tma_lma")
    p.add_argument("--hf-mirror", action="store_true",
                   help="设置 HF_ENDPOINT=hf-mirror（国内下 CLIPSeg）")
    return p.parse_args()


def load_lma(ckpt: str, device: str):
    """加载 LMAAdapter。strict=False：checkpoint 没有 simi_net/sub_impor_net，
    这两个 LMA 新增模块保持随机初始化（本脚本只验证结构跑通）。"""
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt, map_location=device)
    model = LMAAdapter()
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[warn] LMA 新增模块不在 checkpoint，随机初始化：{len(missing)} 个参数")
    if unexpected:
        print(f"[warn] checkpoint 中未被加载的键：{len(unexpected)} 个")
    return model.eval().to(device)


@torch.no_grad()
def main():
    args = parse_args()
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image

    img = to_tensor(Image.open(args.image).convert("RGB"))       # [3,H,W] float [0,1]
    x = img.unsqueeze(0)                                         # [1,3,H,W]
    prompts = args.prompt.split(".") if args.mode == "multiclass" else [args.prompt]

    # ---- 1. TMA：文本 → 掩膜（吴多迅 TextMaskGenerator）----
    gen_kwargs = {"method": args.method, "device": device}
    if args.method == "clipseg":
        gen_kwargs["backend"] = args.clipseg_backend
        gen_kwargs["weights_path"] = args.clipseg_weights
    gen = TextMaskGenerator(**gen_kwargs)
    res = gen.generate(img, prompts, mode=args.mode)
    mask = prepare_cva_mask(res.probs.float(), threshold=args.threshold)  # [1,1,H,W] 0/1
    print(f"[TMA] {len(prompts)} 个 prompt，ROI 覆盖率 {(mask > 0).float().mean():.2%}")

    # ---- 2. LMA：隐空间调制压缩 ----
    model = load_lma(args.gain_ckpt, device)
    x = x.to(device)
    mask = mask.to(device)
    encoded = model.compress_lma(x, mask, args.position)
    x_hat = model.decompress_lma(encoded)["x_hat"].clamp(0, 1)          # [1,3,H,W]

    # ---- 3. 指标（8bit 口径，与 run_tma_cva.py / eval_kodak.py 一致）----
    bits = sum(len(s) * 8 for group in encoded["strings"] for s in group)
    h, w = x.shape[-2:]
    bpp = bits / (h * w)

    x_q = torch.round(x.clamp(0, 1) * 255)
    x_hat_q = torch.round(x_hat.clamp(0, 1) * 255)
    v_psnr = psnr(x_q, x_hat_q)
    v_roi_psnr = roi_psnr(x_q, x_hat_q, mask[0, 0])
    print(f"[result] bpp={bpp:.4f}  PSNR={v_psnr:.3f} dB  ROI-PSNR={v_roi_psnr:.3f} dB")

    # ---- 4. 保存 ----
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    to_pil_image(x_hat[0].cpu()).save(
        os.path.join(args.out, f"{stem}_recon_pos{args.position}.png"))
    to_pil_image(mask[0].cpu()).save(os.path.join(args.out, f"{stem}_mask.png"))
    print(f"[save] -> {args.out}/")


if __name__ == "__main__":
    main()
