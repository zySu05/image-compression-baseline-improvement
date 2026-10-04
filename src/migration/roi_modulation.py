"""迁移原型：ROI 隐式比特分配 + 连续 σ + 可变码率 gain 三控合一。

对应 Customized 代码（third_party/Customizable-ROI-Based-Deep-Image-Compression/model_tic.py）
里的 LMA（隐空间掩膜注意力）+ CVA（σ）机制，并加入可变码率 gain。
这是迁移到可变码率 baseline 的参考实现，接口已对齐"三控独立可调"。

核心公式：
    y = y_base * gain(rate) * (imp(y_base) + simi(mask, σ))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_sigma_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """σ 连续化的掩膜加权（对应 Customized 写死的 where(sim>0.85, 1.0, σ)）。

    原文（写死）: w = where(sim > 0.85, 1.0, 0.01)
    连续化:      w = σ + (1 - σ) * mask_prob       # mask_prob ∈ [0,1] 可微

    Args:
        mask: 文本 ROI 掩膜概率图 [B, 1, H, W]，取值 [0,1]（来自改进 1 的 Grounded-SAM2）。
        sigma: 背景权重，连续可调。σ→1 全图等权，σ→0 只保 ROI。

    Returns:
        w: [B, 1, H, W]，ROI 处 ≈1，背景处 ≈σ。
    """
    return sigma + (1.0 - sigma) * mask


class ROIModulation(nn.Module):
    """隐空间 ROI 调制模块（迁移后的 LMA + 连续 σ + gain）。

    用法（接在可变码率 baseline 的 encoder 输出之后）：

        y_base = encoder(x)                    # 可变码率 baseline 的隐变量
        y = roi_mod(y_base, mask, sigma, rate) # 三控调制
        z = hyper_encoder(y)                   # 之后接超先验 + 熵编码

    三个控制彼此独立：
        rate —— 整体 bpp（gain，连续）
        mask —— 空间/通道级 ROI 调制（来自文本掩膜）
        sigma —— ROI/背景码率分配权衡（连续）
    """

    def __init__(self, num_channels: int, mask_hidden: int = 128):
        super().__init__()
        # 掩膜 1ch -> num_channels ch（3×stride2 下采样，对应 Customized 的 simi_net）
        self.simi_net = nn.Sequential(
            nn.Conv2d(1, mask_hidden // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(mask_hidden // 2, mask_hidden, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(mask_hidden, num_channels, 3, stride=2, padding=1),
        )
        # 内容侧学习的重要性图（对应 Customized 的 sub_impor_net，简化版）
        self.imp_net = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 1),
        )
        # 增益单元（对应 Choi ICCV'19 gain unit，控制整体 bpp）
        self.gain = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(
        self,
        y_base: torch.Tensor,
        mask: torch.Tensor,
        sigma: float = 0.01,
        rate: float = 1.0,
    ) -> torch.Tensor:
        # 1) σ 连续化掩膜加权
        w = continuous_sigma_mask(mask, sigma)

        # 2) 文本掩膜调制（simi_net）
        simi = self.simi_net(w)
        simi = torch.sigmoid(simi)  # [B, C, H/8, W/8]

        # 3) 内容重要性调制（imp_net）
        imp = self.imp_net(y_base)
        imp = torch.tanh(imp)       # [B, C, H/16, W/16]

        # 尺寸对齐：simi 上采样到与 imp 一致
        if simi.shape[-2:] != imp.shape[-2:]:
            simi = F.interpolate(simi, size=imp.shape[-2:], mode="bilinear", align_corners=False)

        # 4) 三控合一：gain(rate) * (内容重要性 + 文本掩膜调制)
        modulation = imp + simi
        y = y_base * (self.gain * rate) * modulation
        return y
