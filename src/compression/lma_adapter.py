"""LMA 接入：把 Customized 的隐空间掩膜注意力（Latent Mask Attention）迁移到可变码率 baseline。

对应 third_party/Customizable-ROI-Based-Deep-Image-Compression/model_tic.py 的 forward：

    y_codec_a6 = g_a6(g_a(x))                              # 最终潜变量 [1,M,H/16,W/16]
    y_soft     = softsign(tanh(sub_impor_net(y_codec)))    # 内容重要性（可学习）
    sim_sig    = sigmoid(simi_net(mask))                   # 文本相似度（可学习）
    y          = y_codec_a6 * (y_soft + sim_sig)           # ← LMA：乘法调制

这里把它接到 CompressAI Hyperprior baseline（bmshj2018_hyperprior）上：

    y = base.g_a(x)                       # baseline 潜变量，等价于上面的 y_codec_a6
    y = y * (y_soft + sim_sig)            # 新增 LMA 调制
    v = y * gain(position)                # 再接武力澎的可变码率 gain（原逻辑不动）

与 CVA（customized_gain_adapter.py）的关键区别：
  - CVA 是「空间增益」，编码前乘、解码时要撤销，所以必须把 mask 塞进码流（mask_png）；
  - LMA 是「调制」，被 g_s 吸收，解码端不撤销，所以 decode 不需要 mask，也不占额外码流。

注意：
  - 本类新增的 simi_net / sub_impor_net 不在现有 gain checkpoint 里，需小规模训练才有意义；
  - 一处刻意简化：原文 sub_impor_net 作用在 g_a 中间特征 [N,H/8,W/8] 上，这里直接作用在
    最终潜变量 [M,H/16,W/16] 上（baseline 的 base.g_a 是整体 Module，拆开要动 base 内部），
    公式一致、只换了重要性图的输入来源。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compression.continuous_gain import ContinuousGainHyperprior


class LMAAdapter(ContinuousGainHyperprior):
    """LMA（隐空间掩膜注意力）+ 可变码率 gain 的最小迁移适配器。

    用法（推理）：
        model = LMAAdapter()
        model.load_state_dict(checkpoint["model"], strict=False)  # simi/sub 随机初始化
        encoded = model.compress_lma(x, mask, position)
        x_hat = model.decompress_lma(encoded)["x_hat"]

    输入：
        x:       [1,3,H,W] 像素 [0,1]（H、W 需能被 16 整除，与 baseline 一致）
        mask:    [1,1,H,W] 文本 ROI 掩膜（0/1 或概率均可，simi_net 自行学习映射）
        position: 0~3 全局 gain 档位（连续可插值）
    """

    def __init__(self, quality=1, mask_hidden=(64, 128)):
        super().__init__(quality=quality)
        M = self.log_gain.shape[1]  # 潜变量通道数，与 gain 对齐（bmshj2018=192）

        # 文本相似度网络：mask [1,1,H/2,W/2] -> 3×stride2 -> [1,M,H/16,W/16]
        # （与 model_tic.py 的 simi_net 结构一致）
        self.simi_net = nn.Sequential(
            nn.Conv2d(1, mask_hidden[0], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(mask_hidden[0], mask_hidden[1], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(mask_hidden[1], M, 3, stride=2, padding=1),
        )

        # 内容重要性网络：y [1,M,H/16,W/16] -> [1,M,H/16,W/16]
        # （对应 model_tic.py 的 sub_impor_net，简化为 stride-1 残差结构）
        self.sub_impor_net = nn.Sequential(
            nn.Conv2d(M, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, M, 1),
        )

        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()

    def lma_modulation(self, y, mask):
        """计算 LMA 调制图 imp + sim，形状 [1,M,H/16,W/16]。"""
        # 掩膜先下采样到原图一半，再走 simi_net（3×stride2），与 model_tic.py 一致
        mask_half = F.interpolate(
            mask, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        sim = torch.sigmoid(self.simi_net(mask_half))

        imp = self.softsign(self.tanh(self.sub_impor_net(y)))

        # 尺寸兜底对齐（奇数边长时可能差 1）
        if sim.shape[-2:] != y.shape[-2:]:
            sim = F.interpolate(
                sim, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
        return imp + sim

    def forward(self, x, mask, position):
        """训练用前向（与 TrainableGainHyperprior.forward 一致，插入 LMA 调制）。"""
        gain = self.gain_at(position)
        y = self.base.g_a(x)
        y = y * self.lma_modulation(y, mask)
        v = y * gain

        z = self.base.h_a(torch.abs(v))
        z_hat, z_likelihoods = self.base.entropy_bottleneck(z)

        scales_hat = self.base.h_s(z_hat)
        v_hat, v_likelihoods = self.base.gaussian_conditional(v, scales_hat)

        y_hat = v_hat / gain
        x_hat = self.base.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": v_likelihoods, "z": z_likelihoods},
        }

    @torch.no_grad()
    def compress_lma(self, x, mask, position):
        self.eval()
        self.base.update(force=True)

        gain = self.gain_at(position)
        x = x.to(gain.device)
        mask = mask.to(gain.device)

        y = self.base.g_a(x)
        y = y * self.lma_modulation(y, mask)
        v = y * gain

        z = self.base.h_a(torch.abs(v))
        z_strings = self.base.entropy_bottleneck.compress(z)
        z_hat = self.base.entropy_bottleneck.decompress(z_strings, z.shape[-2:])

        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(scales)
        v_strings = self.base.gaussian_conditional.compress(v, indexes)

        return {
            "strings": [v_strings, z_strings],
            "shape": z.shape[-2:],
            "position": float(position),
        }

    @torch.no_grad()
    def decompress_lma(self, encoded):
        """解码端不撤销 LMA 调制（被 g_s 吸收），所以不需要 mask。"""
        self.eval()
        self.base.update(force=True)

        gain = self.gain_at(encoded["position"])
        strings = encoded["strings"]

        z_hat = self.base.entropy_bottleneck.decompress(strings[1], encoded["shape"])
        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(scales)
        v_hat = self.base.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )

        y_hat = v_hat / gain
        x_hat = self.base.g_s(y_hat).clamp(0, 1)

        return {"x_hat": x_hat}
