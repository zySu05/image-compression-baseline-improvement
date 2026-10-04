"""改进 3：可变码率（单模型实现不同 bpp）。

- 短期：直接用 CompressAI 的多 quality 模型跑 RD 曲线（多档基准）。
- 目标：接入 gain unit / 多增益 + 量化步长，做到单模型连续 bpp。
详见《改进3-可变码率编码方法调研.md》。
"""
from __future__ import annotations

from typing import Dict, List

import torch


# 架构名 -> compressai.zoo 里的函数名
_MODEL_FNS = {
    "bmshj2018-factorized": "bmshj2018_factorized",
    "bmshj2018-hyperprior": "bmshj2018_hyperprior",
    "mbt2018-mean": "mbt2018_mean",
    "cheng2020-anchor": "cheng2020_anchor",
    "cheng2020-attn": "cheng2020_attn",
}


def load_model(arch: str, quality: int, device: str = "cuda"):
    """加载 CompressAI 预训练模型。"""
    import compressai.zoo as zoo
    if arch not in _MODEL_FNS:
        raise ValueError(f"unknown arch: {arch}，可选 {list(_MODEL_FNS)}")
    fn = getattr(zoo, _MODEL_FNS[arch])
    return fn(quality=quality, pretrained=True).to(device).eval()


def compress_one(model, x: torch.Tensor):
    """x: [1,3,H,W] float [0,1]。返回 (x_hat, strings)。"""
    model.update(force=True)
    with torch.no_grad():
        out_enc = model.compress(x)
    strings = out_enc["strings"]
    out_dec = model.decompress(strings, x.shape[-2:])
    x_hat = out_dec["x_hat"] if isinstance(out_dec, dict) else out_dec
    return x_hat, strings


def bitrate(strings, num_pixels: int) -> float:
    """码流字节数 -> bpp。strings 为 compress 输出的嵌套 list。"""
    total_bits = 0
    for per_image in strings:
        for stream in per_image:
            total_bits += len(stream) * 8
    return total_bits / num_pixels


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2)
    return float(10 * torch.log10(1.0 / mse)) if mse.item() > 0 else float("inf")


def rd_curve(
    arch: str,
    qualities: List[int],
    images: List[torch.Tensor],
    device: str = "cuda",
) -> List[Dict[str, float]]:
    """对一组图像，输出每个 quality 下的 bpp / psnr（RD 曲线数据点）。"""
    rows: List[Dict[str, float]] = []
    for q in qualities:
        model = load_model(arch, q, device)
        for x in images:
            x = x.to(device)
            x_hat, strings = compress_one(model, x)
            num_pixels = x.shape[-2] * x.shape[-1]
            rows.append({
                "quality": float(q),
                "bpp": bitrate(strings, num_pixels),
                "psnr": _psnr(x.cpu(), x_hat.cpu()),
            })
    return rows


class GainUnit(torch.nn.Module):
    """Choi ICCV'19 增益单元（参考实现）：缩放隐变量实现可变码率。

    接入方式：在 encoder 输出 y 处乘 gain，decoder 输入前除 gain；
    rate 为连续调节因子。多增益/量化步长版本见《改进3-可变码率编码方法调研.md》。
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, y: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        return y * (self.gain * rate)
