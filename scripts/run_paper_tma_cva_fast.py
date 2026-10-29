import argparse
import gc
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path.home() / "image-compression-baseline-improvement"
PAPER = ROOT / "third_party/Customizable-ROI-Based-Deep-Image-Compression"
TIC_CKPT = PAPER / "checkpoints/best_loss_0819_0035.pth.tar"

if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_tic import TIC
from src.roi.mask_generator import TextMaskGenerator
from src.migration.tma_integration import prepare_cva_mask

torch.set_num_threads(4)


def load_tic(device: str):
    print("1/3 加载论文 TIC...", flush=True)

    model = TIC(N=128, M=192)

    # 不使用模型内部 LSeg，避免重复占显存
    model.net_lseg = torch.nn.Identity()
    gc.collect()

    ckpt = torch.load(TIC_CKPT, map_location="cpu", weights_only=True)
    source = ckpt["state_dict"]

    mapping = {
        "matrix": "matrices",
        "bias": "biases",
        "factor": "factors",
    }

    translated = {}
    for key, value in source.items():
        match = re.fullmatch(
            r"entropy_bottleneck\._(matrix|bias|factor)(\d+)",
            key,
        )
        if match:
            kind, index = match.groups()
            key = f"entropy_bottleneck.{mapping[kind]}.{index}"
        translated[key] = value

    target = model.state_dict()
    errors = [
        key for key, value in translated.items()
        if key not in target or value.shape != target[key].shape
    ]
    missing = set(target) - set(translated)

    # 只允许缺 net_lseg
    bad_missing = [k for k in missing if not k.startswith("net_lseg.")]
    if errors or bad_missing:
        raise RuntimeError(
            f"TIC 参数不匹配: errors={errors[:5]}, bad_missing={bad_missing[:5]}"
        )

    compatible = {k: v for k, v in translated.items() if k in target}
    model_dict = model.state_dict()
    model_dict.update(compatible)
    model.load_state_dict(model_dict, strict=False)

    del ckpt, source, translated, compatible
    gc.collect()

    model = model.to(device).eval()
    model.update(force=True)

    print("论文 TIC 加载成功", flush=True)
    return model


def prepare_input(image_path: str, device: str):
    with Image.open(image_path) as im:
        original = to_tensor(im.convert("RGB")).unsqueeze(0)

    _, _, h, w = original.shape
    pad_h = (256 - h % 256) % 256
    pad_w = (256 - w % 256) % 256
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    x = F.pad(
        original, (left, right, top, bottom),
        mode="constant", value=0
    ).to(device)

    return original.to(device), x, (top, bottom, left, right)


def make_similarity_from_tma(
    original,
    x,
    pads,
    prompt: str,
    device: str,
    clipseg_backend: str,
    clipseg_weights: str,
    threshold: float,
):
    print("2/3 用 TextMaskGenerator 生成 similarity...", flush=True)

    img = original[0].detach().cpu()   # [3,H,W]

    gen = TextMaskGenerator(
        method="clipseg",
        device=device,
        backend=clipseg_backend,
        weights_path=clipseg_weights,
    )

    res = gen.generate(img, [prompt], mode="multiclass")

    # 先得到原图尺寸的二值掩膜 [1,1,H,W]
    mask = prepare_cva_mask(res.probs.float(), threshold=threshold)

    top, bottom, left, right = pads
    mask = F.pad(
        mask,
        (left, right, top, bottom),
        mode="constant",
        value=0.0,
    ).to(device)

    # 论文 TIC 这里吃的是半分辨率 similarity
    similarity = F.interpolate(
        mask,
        size=(x.shape[-2] // 2, x.shape[-1] // 2),
        mode="bilinear",
        align_corners=False,
    )

    if not torch.isfinite(similarity).all():
        raise RuntimeError("similarity 出现 NaN/Inf")

    print("prompt:", prompt, flush=True)
    print("similarity shape:", tuple(similarity.shape), flush=True)
    print(
        "similarity range:",
        float(similarity.min()),
        float(similarity.max()),
        flush=True,
    )
    print(
        "ROI 覆盖率:",
        f"{float((similarity > 0.85).float().mean()):.2%}",
        flush=True,
    )

    return similarity


def crop_output(x_hat, pads):
    top, bottom, left, right = pads
    h_end = x_hat.shape[-2] - bottom if bottom else x_hat.shape[-2]
    w_end = x_hat.shape[-1] - right if right else x_hat.shape[-1]
    return x_hat[..., top:h_end, left:w_end]


def calc_psnr_8bit(original, recon):
    ref = torch.round(original.clamp(0, 1) * 255)
    rec = torch.round(recon.clamp(0, 1) * 255)
    mse = torch.mean((ref - rec) ** 2).item()
    return float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


def save_image(x_hat, out_path: Path):
    rec = torch.round(x_hat.clamp(0, 1) * 255)
    Image.fromarray(
        rec[0].permute(1, 2, 0).byte().cpu().numpy()
    ).save(out_path)


def main():
    p = argparse.ArgumentParser(description="论文 TIC + 现有 TMA 的快速验证脚本")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--mode", choices=["forward", "bitstream"], default="bitstream")
    p.add_argument("--device", default="auto")
    p.add_argument("--clipseg-backend", default="hf")
    p.add_argument("--clipseg-weights", default="checkpoints/clipseg")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device, flush=True)
    print("image :", args.image, flush=True)

    model = load_tic(device)
    original, x, pads = prepare_input(args.image, device)

    similarity = make_similarity_from_tma(
        original=original,
        x=x,
        pads=pads,
        prompt=args.prompt,
        device=device,
        clipseg_backend=args.clipseg_backend,
        clipseg_weights=args.clipseg_weights,
        threshold=args.threshold,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    print("3/3 开始压缩重建...", flush=True)
    start = time.perf_counter()

    with torch.no_grad():
        if args.mode == "forward":
            result = model(x, similarity)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            x_hat = crop_output(result["x_hat"], pads)

            pixels = original.shape[0] * original.shape[2] * original.shape[3]
            estimated_bpp = sum(
                -torch.log(likelihood.clamp_min(1e-9)).sum().item()
                / (math.log(2) * pixels)
                for likelihood in result["likelihoods"].values()
            )
            actual_bpp = None

        else:
            encode_start = time.perf_counter()
            encoded = model.compress_1(x, similarity)
            if device == "cuda":
                torch.cuda.synchronize()
            encode_elapsed = time.perf_counter() - encode_start

            decode_start = time.perf_counter()
            decoded = model.decompress_1(encoded["strings"], encoded["shape"])
            if device == "cuda":
                torch.cuda.synchronize()
            decode_elapsed = time.perf_counter() - decode_start

            elapsed = time.perf_counter() - start
            x_hat = decoded["x_hat"] if isinstance(decoded, dict) else decoded
            x_hat = crop_output(x_hat, pads)

            bits = sum(len(s[0]) * 8 if isinstance(s, list) else len(s) * 8
                       for group in encoded["strings"] for s in group)
            actual_bpp = bits / (original.shape[-2] * original.shape[-1])
            estimated_bpp = None

    if x_hat.shape != original.shape:
        raise RuntimeError(
            f"重建尺寸不一致: {tuple(x_hat.shape)} vs {tuple(original.shape)}"
        )
    if not torch.isfinite(x_hat).all():
        raise RuntimeError("重建图像出现 NaN/Inf")

    psnr = calc_psnr_8bit(original, x_hat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.image).stem
    out_path = out_dir / f"{stem}_{args.mode}.png"
    save_image(x_hat, out_path)

    print("\n===== 结果 =====")
    print("mode:", args.mode)
    print("total elapsed:", round(elapsed, 4), "s")
    print("recon shape:", tuple(x_hat.shape))
    print("PSNR dB:", round(psnr, 4))
    if estimated_bpp is not None:
        print("estimated BPP:", round(estimated_bpp, 6))
    if actual_bpp is not None:
        print("actual BPP:", round(actual_bpp, 6))
        print("encode elapsed:", round(encode_elapsed, 4), "s")
        print("decode elapsed:", round(decode_elapsed, 4), "s")
    print("saved image:", out_path)


if __name__ == "__main__":
    main()
