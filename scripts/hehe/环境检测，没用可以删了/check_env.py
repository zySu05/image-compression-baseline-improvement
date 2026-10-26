#!/usr/bin/env python
"""环境自检：装完依赖、下完权重后跑一遍，确认每一项都到位。

用法：
  python scripts/check_env.py          # 快速检查（依赖 + 权重文件）
  python scripts/check_env.py --full   # 额外构造 LSeg / TIC 模型（慢，约 1 分钟）

退出码 0 表示全部通过；有 FAIL 时返回 1。
"""
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSEG_REPO = os.path.join(ROOT, "third_party", "Customizable-ROI-Based-Deep-Image-Compression")
CKPT_DIR = os.path.join(LSEG_REPO, "checkpoints")

CLIP_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "clip", "ViT-B-32.pt")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

# 期望的权重文件与精确字节数（用来判断是否下完整）
EXPECTED = {
    "CLIP ViT-B/32": (CLIP_CACHE, 353976522),
    "LSeg ckpt": (os.path.join(CKPT_DIR, "lseg_minimal_e200.ckpt"), 1729204261),
}

PY_PKGS = [
    ("torch", True),
    ("torchvision", True),
    ("numpy", True),
    ("PIL", True),
    ("cv2", True),
    ("yaml", True),
    ("tqdm", True),
    ("compressai", True),
    ("lpips", True),
    ("pytorch_msssim", True),
    ("pandas", True),
    ("timm", True),
    ("transformers", True),
    ("clip", True),
    ("thop", False),   # eval_kodak.py 顶层 import，缺了跑不了官方评测
]

results = []


def check(name, ok, detail="", optional=False):
    """optional=True 时，不通过记为 WARN（不计入 FAIL）。"""
    status = "PASS" if ok else ("WARN" if optional else "FAIL")
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  — {detail}" if detail else ""))


def check_python():
    v = sys.version.split()[0]
    check("Python 版本", sys.version_info[:2] == (3, 10), f"{v}（建议 3.10.x）")


def check_pkgs():
    for mod, required in PY_PKGS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            check(f"import {mod}", True, str(ver))
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, f"{type(e).__name__}: {e}", optional=not required)


def check_torch():
    try:
        import torch

        check("CUDA 可用", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到 GPU")
        check("torch CUDA 版本", torch.version.cuda is not None, str(torch.version.cuda))
    except Exception as e:  # noqa: BLE001
        check("torch 检查", False, str(e))


def check_weights():
    for name, (path, size) in EXPECTED.items():
        if not os.path.isfile(path):
            check(name, False, f"缺失：{path}")
            continue
        actual = os.path.getsize(path)
        check(name, actual == size, f"{actual:,} B / 期望 {size:,} B — {path}")

    # 压缩模型权重（文件名随下载版本变化，只看有没有、大小是否合理）
    cands = []
    if os.path.isdir(CKPT_DIR):
        cands = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth.tar")]
    if cands:
        for f in cands:
            p = os.path.join(CKPT_DIR, f)
            mb = os.path.getsize(p) / 1e6
            check(f"压缩模型权重 {f}", mb > 50, f"{mb:.1f} MB")
    else:
        check("压缩模型权重", False, f"{CKPT_DIR} 下没有 .pth.tar")

    # CLIPSeg 走 HuggingFace 缓存
    cached = os.path.isdir(HF_CACHE) and any(
        d.startswith("models--CIDAS--clipseg") for d in os.listdir(HF_CACHE)
    )
    check("CLIPSeg CIDAS/clipseg-rd64-refined", cached,
          "已缓存" if cached else "未缓存，首次运行会自动下载（需 HF_ENDPOINT=https://hf-mirror.com）")


def check_models():
    """--full：真正把模型建起来，验证版本兼容性。"""
    sys.path.insert(0, LSEG_REPO)
    import timm

    _orig = timm.create_model

    def _no_pretrain(*a, **k):
        # LSeg 的 ckpt 已包含全部骨干权重，跳过 1.2GB ImageNet ViT-L 下载
        k["pretrained"] = False
        return _orig(*a, **k)

    timm.create_model = _no_pretrain
    try:
        import torch
        from lseg.lseg_net import LSegNet

        net = LSegNet(backbone="clip_vitl16_384", features=256, crop_size=256,
                      arch_option=0, block_depth=0, activation="lrelu")
        state = torch.load(EXPECTED["LSeg ckpt"][0], map_location="cpu", weights_only=False)
        net.load_state_dict(state, strict=True)
        check("LSegNet 构建 + ckpt strict 加载", True, "timm 版本兼容")
    except Exception as e:  # noqa: BLE001
        check("LSegNet 构建 + ckpt strict 加载", False, f"{type(e).__name__}: {e}")

    try:
        from model_tic import TIC

        m = TIC()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        check("TIC 模型构建", True, f"{n:.2f}M 参数（compressai 兼容）")
    except Exception as e:  # noqa: BLE001
        check("TIC 模型构建", False, f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="额外构建模型（慢）")
    args = p.parse_args()

    print(f"项目根目录: {ROOT}\n")
    check_python()
    check_torch()
    check_pkgs()
    check_weights()
    if args.full:
        print("\n--- 构建模型（较慢）---")
        check_models()

    failed = [r for r in results if r[1] == "FAIL"]
    warned = [r for r in results if r[1] == "WARN"]
    print(f"\n{'=' * 50}\n{len(results) - len(failed) - len(warned)}/{len(results)} 项通过"
          f"（{len(warned)} 项警告，{len(failed)} 项失败）")
    for label, group in (("警告", warned), ("未通过", failed)):
        if group:
            print(f"{label}：")
            for name, _, detail in group:
                print(f"  - {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
