"""TMA 接入胶水：把改进 1 的文本掩膜生成，接进 Customized 的压缩管线。

职责：把「任意文本掩膜生成器」输出的软掩膜，转成 Customized `TIC.forward(x, similarity)`
所期望的 similarity 输入。这里只做格式/分辨率/阈值转换，不碰模型本身，
所以本模块无重依赖（不 import model_tic / lseg / clip），可在任何机器上单独测试。

与论文/代码的对应关系（见《Customized代码解读.md》《baseline算法介绍.md》）：
  - TMA（文本→掩膜）      ：由 src/roi/mask_generator.py 完成（CLIPSeg 现在可用，
                            Grounded-SAM2 接口已留，补实现后原样替换）。
  - CVA（ROI/背景加权 σ） ：原文在 model_tic.py 里写死 `where(sim>0.85, 1.0, 0.01)`。
                            这里把「二值化/σ 加权」提前到掩膜侧完成，喂给模型时
                            只传 {1.0, σ} 两个值，正好能穿透模型内部的 0.85 阈值。
  - LMA（隐空间调制）     ：模型内部 simi_net + sub_impor_net，本模块不涉及。

分辨率约定（关键，已从 model_tic.py 推导确认）：
  - forward 的 similarity 必须是「输入图像 padding 后尺寸的一半」：即 [1,1,PH/2,PW/2]。
    原因：simi_net 是 3 次 stride2（÷8），要落到 H/16 与 y_codec_a6（g_a 的 ÷8 再 ÷2）对齐。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_padding(in_h: int, in_w: int, min_div: int = 256):
    """与 eval_kodak.py:39 一致：对称 pad 到 min_div 的整数倍。

    Returns:
        pad: (left, right, top, bottom)，供 F.pad 使用。
        out_h, out_w: padding 后的高/宽。
    """
    out_h = (in_h + min_div - 1) // min_div * min_div
    out_w = (in_w + min_div - 1) // min_div * min_div
    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top
    return (left, right, top, bottom), out_h, out_w


def merge_masks(probs: torch.Tensor) -> torch.Tensor:
    """把 N 个类别的软掩膜合并成一张 [H,W]。

    probs: [N,H,W]，每个元素 ∈[0,1]（来自 MaskResult.probs）。
    多类别取 max（像素属于任一 ROI 即视为 ROI）；句子模式 N=1 直接取。
    """
    if probs.dim() == 2:
        return probs
    if probs.shape[0] == 1:
        return probs[0]
    return probs.max(dim=0).values


def prepare_similarity(
    mask: torch.Tensor,
    pad,
    sigma: float = 0.01,
    threshold: float = 0.5,
) -> torch.Tensor:
    """把 [H,W] 软掩膜转成 TIC.forward 所需的 similarity [1,1,PH/2,PW/2]。

    步骤：
      1) clamp 到 [0,1] → pad 到与输入图像一致的 PH×PW（掩膜 ROI 与图像逐像素对齐）；
      2) 双线性下采样到 PH/2×PW/2（这是 TIC forward 期望的掩膜分辨率）；
      3) 二值化/σ 加权：ROI=1.0，背景=σ（默认 0.01，与论文 CVA 一致）。

    Args:
        mask: [H,W] 软掩膜（CLIPSeg 是 sigmoid 概率；将来 Grounded-SAM2 是 0/1 或概率）。
        pad: compute_padding 返回的 (left, right, top, bottom)。
        sigma: 背景失真权重（CVA 的 σ）。ROI 恒为 1.0，背景为 σ。
        threshold: 掩膜二值化阈值。CLIPSeg 概率默认 0.5；
                   （论文原版 LSeg 是余弦相似度、阈值 0.85，二者语义不同，见脚本说明）。

    Returns:
        similarity: [1,1,PH/2,PW/2]，取值 {1.0, σ}，可直接喂给 model.forward(x, similarity)。

    注意：这里输出 {1.0, σ} 而非连续值，是为了能「穿透」model_tic.py 里写死的
    `where(sim>0.85, 1.0, 0.01)`（1.0→1.0，σ=0.01→0.01，二者都不越过 0.85）。
    σ 连续化（w = σ + (1-σ)·prob）需要同时改模型内部那行，属于 CVA 迁移，另行处理。
    """
    w = mask.float().clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    w = F.pad(w, pad, mode="constant", value=0.0)              # [1,1,PH,PW]
    ph, pw = w.shape[-2], w.shape[-1]
    # 先对软掩膜下采样，再二值化，避免「先二值化后双线性」造成的边缘侵蚀
    w = F.interpolate(w, size=(ph // 2, pw // 2), mode="bilinear", align_corners=False)
    w = torch.where(
        w > threshold,
        torch.tensor(1.0, device=w.device, dtype=w.dtype),
        torch.tensor(sigma, device=w.device, dtype=w.dtype),
    )
    return w  # [1,1,PH/2,PW/2]
