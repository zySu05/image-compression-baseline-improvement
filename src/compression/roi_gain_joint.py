import io
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from src.compression.continuous_gain import ContinuousGainHyperprior


class ROIGainJoint(ContinuousGainHyperprior):
    """
    最小联合原型：
    position 控制全局 Gain；
    sigma 控制 ROI/背景的空间调制。
    不是 Customized CVA 的正式复现。
    """

    def spatial_gain(self, mask, y_shape, position, sigma):
        sigma = float(sigma)
        if not math.isfinite(sigma):
            raise ValueError("sigma必须为有限值")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask形状必须为[B,1,H,W]")

        m = F.interpolate(
            mask.float(),
            size=y_shape[-2:],
            mode="nearest",
        )
        global_gain = self.gain_at(position)

        # ROI 增益提高，背景增益降低
        spatial = torch.exp(sigma * (2 * m - 1))
        return global_gain * spatial

    @staticmethod
    def pack_mask(mask):
        im = to_pil_image(mask[0].cpu())
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def unpack_mask(data, device):
        with Image.open(io.BytesIO(data)) as im:
            mask = to_tensor(im.convert("L")).unsqueeze(0)
        return (mask >= 0.5).float().to(device)

    @torch.no_grad()
    def compress_joint(self, x, mask, position, sigma):
        self.eval()
        self.base.update(force=True)

        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("当前仅支持单张图像")
        if mask.shape != (1, 1, *x.shape[-2:]):
            raise ValueError("ROI掩膜尺寸与图像不匹配")

        device = self.log_gain.device
        x = x.to(device)
        mask = (mask >= 0.5).float().to(device)

        y = self.base.g_a(x)
        gain = self.spatial_gain(
            mask, y.shape, position, sigma
        )
        v = y * gain

        z = self.base.h_a(torch.abs(v))
        z_strings = self.base.entropy_bottleneck.compress(z)
        z_hat = self.base.entropy_bottleneck.decompress(
            z_strings, z.shape[-2:]
        )

        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(scales)
        v_strings = self.base.gaussian_conditional.compress(v, indexes)

        return {
            "strings": [v_strings, z_strings],
            "shape": z.shape[-2:],
            "position": float(position),
            "sigma": float(sigma),
            "mask_png": self.pack_mask(mask),
        }

    @torch.no_grad()
    def decompress_joint(self, encoded):
        self.eval()
        self.base.update(force=True)

        device = self.log_gain.device
        mask = self.unpack_mask(encoded["mask_png"], device)

        strings = encoded["strings"]
        z_hat = self.base.entropy_bottleneck.decompress(
            strings[1], encoded["shape"]
        )
        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(scales)
        v_hat = self.base.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )

        gain = self.spatial_gain(
            mask,
            v_hat.shape,
            encoded["position"],
            encoded["sigma"],
        )
        x_hat = self.base.g_s(v_hat / gain).clamp(0, 1)
        return {"x_hat": x_hat}
