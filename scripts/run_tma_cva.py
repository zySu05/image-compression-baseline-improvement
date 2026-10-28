#!/usr/bin/env python
"""TMA → CVA 对接：文本 → 掩膜（改进1）→ σ 联合压缩（改进3 的 CVA 接口）。

把吴多迅的 TextMaskGenerator（TMA，CLIPSeg）接到武力澎的 CustomizedGainAdapter（CVA σ 接口），
替换 scripts/run_joint_pipeline.py 里内联的 CLIPSeg，统一走 src/roi/mask_generator.py。

链路：
    文本 prompt → TextMaskGenerator → MaskResult.probs [N,H,W]
               → prepare_cva_mask → mask [1,1,H,W]
               → CustomizedGainAdapter.compress_joint(x, mask, position, σ) → 编码
               → decompress_joint → x_hat → bpp/PSNR/ROI-PSNR

依赖（有显卡的同学运行）：
    torch(CUDA)  compressai  transformers  numpy  pillow
    + checkpoints/gain/gain_small_trained.pth（武力澎的 Gain 权重，已在仓库）
    + CLIPSeg 权重（hf 后端自动下 / 或 --clipseg-backend local）

用法：
    python scripts/run_tma_cva.py --image data/kodak/kodim01.png --prompt "a building" \
        --position 1.5 --sigma 0.3 --out results/tma_cva

    # 多类别
    python scripts/run_tma_cva.py --image img.jpg --prompt "person.dog" \
        --position 1.5 --sigma 0.3

σ 语义 = CVA 的「背景权重」：ROI 恒为 1，背景为 σ（σ∈(0,1]，越小越只保 ROI）。
这里用武力澎 CustomizedGainAdapter 的二值 ROI/背景 σ 调制（对应论文 CVA 的
where(sim>0.85, 1.0, σ)）；连续 σ 见 src/migration/roi_modulation.py，属后续正式迁移。
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
from src.compression.customized_gain_adapter import CustomizedGainAdapter  # noqa: E402
from src.evaluation.metrics import psnr, roi_psnr                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TMA→CVA 对接（文本→掩膜→σ 联合压缩）")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="多类别用点分隔；句子直接写")
    p.add_argument("--mode", default="multiclass", choices=["multiclass", "sentence"])
    p.add_argument("--method", default="clipseg", choices=["clipseg", "grounded_sam2"],
                   help="掩膜生成方法；grounded_sam2 接口已留、实现待补")
    p.add_argument("--clipseg-backend", default="hf", choices=["hf", "local"],
                   help="CLIPSeg 后端：hf=HuggingFace 自动下权重，local=本地权重")
    p.add_argument("--clipseg-weights", default=None, help="local 后端时必填")

    p.add_argument("--gain-ckpt", default="checkpoints/gain/gain_small_trained.pth",
                   help="武力澎的 Gain checkpoint（含 CustomizedGainAdapter 权重）")
    p.add_argument("--position", type=float, default=1.5, help="全局 Gain 档位 0~3")
    p.add_argument("--sigma", type=float, default=0.3, help="CVA 背景权重 σ∈(0,1]")
    p.add_argument("--threshold", type=float, default=0.5, help="掩膜二值化阈值")

    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--out", default="results/tma_cva")
    p.add_argument("--hf-mirror", action="store_true",
                   help="设置 HF_ENDPOINT=hf-mirror（国内下 CLIPSeg）")
    return p.parse_args()


def load_cva(ckpt: str, device: str):
    """加载 CustomizedGainAdapter（与 test_customized_gain_adapter.py 一致）。"""
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt, map_location=device)
    model = CustomizedGainAdapter()
    model.load_state_dict(state["model"])
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

    # ---- 2. CVA：σ 联合压缩（武力澎 CustomizedGainAdapter）----
    model = load_cva(args.gain_ckpt, device)
    x = x.to(device)
    mask = mask.to(device)
    encoded = model.compress_joint(x, mask, args.position, args.sigma)
    x_hat = model.decompress_joint(encoded)["x_hat"].clamp(0, 1)          # [1,3,H,W]

    # ---- 3. 指标（8bit 口径，与 run_customized_tma.py / eval_kodak.py 一致）----
    bits = sum(len(s) * 8 for group in encoded["strings"] for s in group)
    h, w = x.shape[-2:]
    latent_bpp = bits / (h * w)
    mask_bpp = len(encoded["mask_png"]) * 8 / (h * w)

    x_q = torch.round(x.clamp(0, 1) * 255)
    x_hat_q = torch.round(x_hat.clamp(0, 1) * 255)
    v_psnr = psnr(x_q, x_hat_q)
    v_roi_psnr = roi_psnr(x_q, x_hat_q, mask[0, 0])
    print(f"[result] 潜变量bpp={latent_bpp:.4f}  掩膜bpp={mask_bpp:.4f}  "
          f"总bpp={latent_bpp + mask_bpp:.4f}  PSNR={v_psnr:.3f} dB  ROI-PSNR={v_roi_psnr:.3f} dB")

    # ---- 4. 保存 ----
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    to_pil_image(x_hat[0].cpu()).save(
        os.path.join(args.out, f"{stem}_recon_pos{args.position}_sigma{args.sigma}.png"))
    to_pil_image(mask[0].cpu()).save(os.path.join(args.out, f"{stem}_mask.png"))
    print(f"[save] -> {args.out}/")


if __name__ == "__main__":
    main()
