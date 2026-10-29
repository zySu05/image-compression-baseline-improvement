import argparse
import gc
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import clip

from PIL import Image
from torchvision.transforms.functional import to_tensor

from model_tic import TIC
from lseg.lseg_net import LSegNet


ROOT = Path.home() / "image-compression-baseline-improvement"
PAPER = ROOT / "third_party/Customizable-ROI-Based-Deep-Image-Compression"
TIC_CKPT = PAPER / "checkpoints/best_loss_0819_0035.pth.tar"
LSEG_CKPT = PAPER / "checkpoints/lseg_minimal_e200.ckpt"
IMAGE = ROOT / "data/kodak/kodim01.png"
OUT = ROOT / "results/paper_cva_lma_speed/paper_lseg"

torch.set_num_threads(4)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



def load_models(device):
    print("1/4 初始化 TIC...", flush=True)

    model = TIC(N=128, M=192)

    # forward 接收外部 similarity，不调用 model.net_lseg。
    # 仅在本次验证中移除未使用的内部 LSeg 副本，避免重复占用显存。
    model.net_lseg = torch.nn.Identity()
    gc.collect()

    ckpt = torch.load(
        TIC_CKPT, map_location="cpu", weights_only=True
    )
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
            key = (
                f"entropy_bottleneck."
                f"{mapping[kind]}.{index}"
            )
        translated[key] = value

    target = model.state_dict()

    errors = [
        key for key, value in translated.items()
        if key not in target
        or value.shape != target[key].shape
    ]

    missing = set(target) - set(translated)

    if errors or missing:
        raise RuntimeError(
            f"TIC 参数不匹配：errors={errors[:5]}, "
            f"missing={sorted(missing)[:5]}"
        )

    model.load_state_dict(translated, strict=True)
    del ckpt, source, translated
    gc.collect()

    model = model.to(device).eval()
    model.update(force=True)

    print("TIC 参数加载与熵模型更新完成", flush=True)

    print("2/4 加载独立 LSeg...", flush=True)

    lseg = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=256,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )

    state = torch.load(
        LSEG_CKPT, map_location="cpu", weights_only=True
    )
    lseg.load_state_dict(state, strict=True)
    del state
    gc.collect()

    lseg = lseg.to(device).eval()

    print("LSeg 严格加载成功", flush=True)
    return model, lseg


def prepare_input(device):
    with Image.open(IMAGE) as im:
        original = to_tensor(im.convert("RGB")).unsqueeze(0)

    _, _, h, w = original.shape

    # 与论文评测一致：对齐至 256 的倍数
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


def make_similarity(lseg, x, prompt="building"):
    print("3/4 计算文本与图像相似度...", flush=True)

    with torch.no_grad():
        image_features = F.normalize(lseg.forward(x), dim=1)

        tokens = clip.tokenize([prompt]).to(x.device)
        text_features = F.normalize(
            lseg.clip_pretrained.encode_text(tokens), dim=1
        )

        similarity = F.cosine_similarity(
            image_features,
            text_features[:, :, None, None],
            dim=1,
        ).unsqueeze(1)

    if not torch.isfinite(similarity).all():
        raise RuntimeError("similarity 出现 NaN 或 Inf")

    print("相似度图形状:", tuple(similarity.shape), flush=True)
    print(
        "相似度范围:",
        float(similarity.min()),
        float(similarity.max()),
        flush=True,
    )

    return similarity


def tma_generate_similarity(lseg, x, prompt="building"):
    """TMA：根据图像和文本提示词生成相似度图。"""
    return make_similarity(lseg, x, prompt)


def cva_lma_encode(model, x, similarity):
    """CVA + LMA：执行论文 TIC 的真实比特流编码。"""
    return model.compress_1(x, similarity)


def cva_lma_decode(model, encoded):
    """对编码结果执行解码，返回模型输出。"""
    return model.decompress_1(
        encoded["strings"], encoded["shape"]
    )


def tma_cva_lma_encode(model, lseg, x, prompt="building"):
    """完整编码接口：TMA 相似度生成 → CVA → LMA → 熵编码。"""
    similarity = tma_generate_similarity(lseg, x, prompt)
    encoded = cva_lma_encode(model, x, similarity)
    return similarity, encoded


def crop_output(x_hat, pads):
    top, bottom, left, right = pads
    h_end = x_hat.shape[-2] - bottom if bottom else x_hat.shape[-2]
    w_end = x_hat.shape[-1] - right if right else x_hat.shape[-1]
    return x_hat[..., top:h_end, left:w_end]


def report(original, x_hat, elapsed, mode, estimated_bpp=None,
           actual_bpp=None):
    if isinstance(x_hat, dict):
        x_hat = x_hat["x_hat"]

    if x_hat.shape != original.shape:
        raise RuntimeError(
            f"重建尺寸不一致：{tuple(x_hat.shape)} "
            f"vs {tuple(original.shape)}"
        )

    if not torch.isfinite(x_hat).all():
        raise RuntimeError("重建图像出现 NaN 或 Inf")

    ref = torch.round(original.clamp(0, 1) * 255)
    rec = torch.round(x_hat.clamp(0, 1) * 255)
    mse = torch.mean((ref - rec) ** 2).item()
    psnr = float("inf") if mse == 0 else 20 * math.log10(
        255 / math.sqrt(mse)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"kodim01_{mode}.png"

    Image.fromarray(
        rec[0].permute(1, 2, 0).byte().cpu().numpy()
    ).save(path)

    print("\n===== 结果 =====")
    print("模式:", mode)
    print("耗时秒:", round(elapsed, 4))
    print("重建形状:", tuple(x_hat.shape))
    print("PSNR dB:", round(psnr, 4))

    if estimated_bpp is not None:
        print("似然估计 BPP:", round(estimated_bpp, 6))

    if actual_bpp is not None:
        print("真实比特流 BPP:", round(actual_bpp, 6))

    print("重建图:", path, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="论文原生 TMA-CVA-LMA 编解码验证"
    )
    parser.add_argument(
        "mode", choices=("forward", "bitstream")
    )
    parser.add_argument(
        "--prompt", default="building",
        help="用于生成文本相似度图的提示词"
    )
    args = parser.parse_args()
    mode = args.mode
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("设备:", device)
    print("图像:", IMAGE)

    model, lseg = load_models(device)
    original, x, pads = prepare_input(device)
    similarity = tma_generate_similarity(lseg, x, args.prompt)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        if mode == "forward":
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

            report(
                original, x_hat, elapsed,
                mode, estimated_bpp=estimated_bpp
            )

        else:
            encoded = cva_lma_encode(model, x, similarity)

            import hashlib
            print("===== 编码流指纹 =====", flush=True)
            for gi, group in enumerate(encoded["strings"]):
                for si, stream in enumerate(group):
                    if isinstance(stream, (bytes, bytearray)):
                        payload = bytes(stream)
                    elif isinstance(stream, (list, tuple)) and all(
                        isinstance(v, (bytes, bytearray))
                        for v in stream
                    ):
                        payload = b"".join(bytes(v) for v in stream)
                    else:
                        payload = repr(stream).encode("utf-8")

                    digest = hashlib.sha256(payload).hexdigest()
                    print(
                        f"group={gi} stream={si} "
                        f"bytes={len(payload)} sha256={digest}",
                        flush=True,
                    )

            if device == "cuda":
                torch.cuda.synchronize()

            encode_elapsed = time.perf_counter() - start

            decode_start = time.perf_counter()
            decoded = cva_lma_decode(model, encoded)
            if device == "cuda":
                torch.cuda.synchronize()
            decode_elapsed = time.perf_counter() - decode_start
            x_hat = decoded["x_hat"] if isinstance(decoded, dict) else decoded
            x_hat = crop_output(x_hat, pads)

            bits = sum(
                len(stream) * 8
                for group in encoded["strings"]
                for stream in group
            )
            pixels = original.shape[0] * original.shape[2] * original.shape[3]
            actual_bpp = bits / pixels

            report(
                original, x_hat,
                encode_elapsed + decode_elapsed,
                mode, actual_bpp=actual_bpp
            )

            print("编码耗时秒:", round(encode_elapsed, 4))
            print("解码耗时秒:", round(decode_elapsed, 4))


if __name__ == "__main__":
    main()
