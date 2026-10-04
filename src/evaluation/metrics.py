"""统一评测指标：PSNR / MS-SSIM / LPIPS / ROI-PSNR / bpp。

约定：图像统一为 [C, H, W] 的 float32 tensor，取值范围 [0, 1]。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _as_3chw(x) -> torch.Tensor:
    """把 HxWx3 uint8 / [1,3,H,W] / [3,H,W] 统一成 [3,H,W] float [0,1]。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"unsupported type: {type(x)}")
    x = x.detach().float()
    if x.dim() == 4:
        x = x.squeeze(0)
    if x.dim() == 3 and x.shape[-1] == 3:  # HxWx3
        x = x.permute(2, 0, 1)
    assert x.dim() == 3 and x.shape[0] == 3, f"expect [3,H,W], got {tuple(x.shape)}"
    if x.max() > 1.0:
        x = x / 255.0
    return x


def psnr(ref, rec, data_range: float = 1.0) -> float:
    ref = _as_3chw(ref)
    rec = _as_3chw(rec)
    mse = F.mse_loss(rec, ref)
    if mse.item() == 0:
        return float("inf")
    return float(10 * torch.log10(data_range ** 2 / mse))


def ms_ssim(ref, rec, data_range: float = 1.0) -> float:
    try:
        from pytorch_msssim import ms_ssim as _ms_ssim
    except ImportError as e:
        raise ImportError("需要 pytorch-msssim：pip install pytorch-msssim") from e
    ref = _as_3chw(ref).unsqueeze(0)
    rec = _as_3chw(rec).unsqueeze(0)
    return float(_ms_ssim(ref, rec, data_range=data_range))


_lpips_fn = None


def lpips(ref, rec) -> float:
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False)
    ref = _as_3chw(ref).unsqueeze(0) * 2.0 - 1.0  # lpips 期望 [-1, 1]
    rec = _as_3chw(rec).unsqueeze(0) * 2.0 - 1.0
    return float(_lpips_fn(ref, rec).item())


def roi_psnr(ref, rec, mask, data_range: float = 1.0) -> float:
    """仅计算 mask 内区域的 PSNR。mask: [H,W] 或 [1,H,W]，0/1。"""
    ref = _as_3chw(ref)
    rec = _as_3chw(rec)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach()
    mask = mask.squeeze().to(torch.bool)
    if mask.dim() == 3:
        mask = mask[0]
    m3 = mask.unsqueeze(0).expand_as(ref)  # [3,H,W]
    ref_m = ref[m3]
    rec_m = rec[m3]
    if ref_m.numel() == 0:
        return float("nan")
    mse = F.mse_loss(rec_m, ref_m)
    if mse.item() == 0:
        return float("inf")
    return float(10 * torch.log10(data_range ** 2 / mse))


def bpp(num_bytes: int, height: int, width: int) -> float:
    """每像素比特数（bits per pixel）。"""
    return num_bytes * 8.0 / (height * width)


def evaluate(
    ref,
    rec,
    mask=None,
    num_bytes: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """一站式评测，返回 dict。size: (height, width)。"""
    ref_t = _as_3chw(ref)
    h, w = size if size is not None else (ref_t.shape[-2], ref_t.shape[-1])
    out: Dict[str, float] = {
        "psnr": psnr(ref_t, rec),
        "ms_ssim": ms_ssim(ref_t, rec),
        "lpips": lpips(ref_t, rec),
    }
    if mask is not None:
        out["roi_psnr"] = roi_psnr(ref_t, rec, mask)
    if num_bytes is not None:
        out["bpp"] = bpp(num_bytes, h, w)
    return out
