import math
import torch
from torch import nn
from compressai.zoo import bmshj2018_hyperprior


class TrainableGainHyperprior(nn.Module):
    def __init__(self, quality=1):
        super().__init__()

        self.base = bmshj2018_hyperprior(
            quality=quality, pretrained=True
        )

        # 第一阶段只训练 Gain，冻结原模型
        for p in self.base.parameters():
            p.requires_grad_(False)

        # 四个初始增益档位
        initial_gains = [0.5, 0.75, 1.0, 1.5]
        self.log_gain = nn.Parameter(
            torch.log(torch.tensor(initial_gains))
            .view(4, 1)
            .expand(4, 192)
            .clone()
        )

    def forward(self, x, level):
        if not 0 <= level < 4:
            raise ValueError("level 必须在0到3之间")

        gain = torch.exp(
            self.log_gain[level]
        ).view(1, 192, 1, 1)

        # 编码器与增益
        y = self.base.g_a(x)
        v = y * gain

        # 与调制后潜变量一致的超先验
        z = self.base.h_a(torch.abs(v))
        z_hat, z_likelihoods = self.base.entropy_bottleneck(z)

        scales_hat = self.base.h_s(z_hat)
        v_hat, v_likelihoods = self.base.gaussian_conditional(
            v, scales_hat
        )

        # 逆增益与图像重建
        y_hat = v_hat / gain
        x_hat = self.base.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": v_likelihoods,
                "z": z_likelihoods,
            }
        }



    @torch.no_grad()
    def compress(self, x, level):
        if not 0 <= level < 4:
            raise ValueError("level 必须在0到3之间")

        self.eval()
        self.base.update(force=True)

        gain = torch.exp(
            self.log_gain[level]
        ).view(1, 192, 1, 1)

        y = self.base.g_a(x)
        v = y * gain

        z = self.base.h_a(torch.abs(v))
        z_strings = self.base.entropy_bottleneck.compress(z)

        z_hat = self.base.entropy_bottleneck.decompress(
            z_strings, z.shape[-2:]
        )

        scales_hat = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(
            scales_hat
        )

        v_strings = self.base.gaussian_conditional.compress(
            v, indexes
        )

        return {
            "strings": [v_strings, z_strings],
            "shape": z.shape[-2:],
            "level": int(level),
        }

    @torch.no_grad()
    def decompress(self, encoded):
        self.eval()
        self.base.update(force=True)

        level = encoded["level"]
        if not 0 <= level < 4:
            raise ValueError("level 必须在0到3之间")

        gain = torch.exp(
            self.log_gain[level]
        ).view(1, 192, 1, 1)

        strings = encoded["strings"]
        shape = encoded["shape"]

        z_hat = self.base.entropy_bottleneck.decompress(
            strings[1], shape
        )

        scales_hat = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(
            scales_hat
        )

        v_hat = self.base.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )

        y_hat = v_hat / gain
        x_hat = self.base.g_s(y_hat).clamp(0, 1)

        return {"x_hat": x_hat}

def rate_distortion_loss(output, target, lmbda):
    pixels = target.shape[0] * target.shape[2] * target.shape[3]

    rate = sum(
        torch.log2(likelihood.clamp_min(1e-9)).sum()
        for likelihood in output["likelihoods"].values()
    )
    bpp = -rate / pixels

    mse = torch.mean((output["x_hat"] - target) ** 2)

    # 与 CompressAI 常见 MSE 训练目标保持一致
    loss = bpp + lmbda * (255 ** 2) * mse

    return loss, bpp, mse
