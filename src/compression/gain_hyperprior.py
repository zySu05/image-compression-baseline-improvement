import torch
from compressai.zoo import bmshj2018_hyperprior


class GainHyperprior:
    """
    固定预训练 Hyperprior 权重的增益实验原型。

    注意：这是机制验证，不是正式训练完成的可变码率模型。
    """

    def __init__(self, quality=1, device="cpu"):
        self.device = torch.device(device)
        self.model = bmshj2018_hyperprior(
            quality=quality,
            pretrained=True
        ).to(self.device).eval()
        self.model.update(force=True)

    @torch.no_grad()
    def compress(self, x, gain=1.0):
        if gain <= 0:
            raise ValueError("gain 必须大于 0")

        model = self.model
        x = x.to(self.device)

        # 1. 主编码器提取特征
        y = model.g_a(x)

        # 2. 对潜变量施加增益
        v = y * gain

        # 3. 基于调制后的潜变量构建超先验
        z = model.h_a(torch.abs(v))
        z_strings = model.entropy_bottleneck.compress(z)
        z_hat = model.entropy_bottleneck.decompress(
            z_strings, z.shape[-2:]
        )

        # 4. 生成熵编码索引并压缩调制后的潜变量
        scales_hat = model.h_s(z_hat)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)
        v_strings = model.gaussian_conditional.compress(v, indexes)

        return {
            "strings": [v_strings, z_strings],
            "shape": z.shape[-2:],
            "gain": float(gain),
        }

    @torch.no_grad()
    def decompress(self, encoded):
        model = self.model
        strings = encoded["strings"]
        shape = encoded["shape"]
        gain = encoded["gain"]

        if gain <= 0:
            raise ValueError("gain 必须大于 0")

        # 1. 解码超先验
        z_hat = model.entropy_bottleneck.decompress(
            strings[1], shape
        )

        # 2. 恢复主潜变量
        scales_hat = model.h_s(z_hat)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)

        v_hat = model.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )

        # 3. 逆增益恢复潜变量尺度
        y_hat = v_hat / gain

        # 4. 重建图像
        x_hat = model.g_s(y_hat).clamp(0, 1)

        return {"x_hat": x_hat}

    @staticmethod
    def count_bits(encoded):
        return sum(
            len(stream) * 8
            for group in encoded["strings"]
            for stream in group
        )
