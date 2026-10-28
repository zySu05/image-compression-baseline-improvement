import math
import torch

from src.compression.trainable_gain_hyperprior import (
    TrainableGainHyperprior,
)


class ContinuousGainHyperprior(TrainableGainHyperprior):

    def gain_at(self, position):
        position = float(position)

        if not math.isfinite(position):
            raise ValueError("position 必须是有限数")

        levels = self.log_gain.shape[0]

        if not 0 <= position <= levels - 1:
            raise ValueError(
                f"position 必须在 0 到 {levels - 1} 之间"
            )

        lo = min(int(math.floor(position)), levels - 1)
        hi = min(lo + 1, levels - 1)
        alpha = position - lo

        log_gain = (
            (1 - alpha) * self.log_gain[lo]
            + alpha * self.log_gain[hi]
        )

        return torch.exp(log_gain).view(1, -1, 1, 1)

    @torch.no_grad()
    def compress_at(self, x, position):
        self.eval()
        self.base.update(force=True)

        gain = self.gain_at(position)
        x = x.to(gain.device)

        y = self.base.g_a(x)
        v = y * gain

        z = self.base.h_a(torch.abs(v))
        z_strings = self.base.entropy_bottleneck.compress(z)
        z_hat = self.base.entropy_bottleneck.decompress(
            z_strings, z.shape[-2:]
        )

        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(
            scales
        )

        v_strings = self.base.gaussian_conditional.compress(
            v, indexes
        )

        return {
            "strings": [v_strings, z_strings],
            "shape": z.shape[-2:],
            "position": float(position),
        }

    @torch.no_grad()
    def decompress_at(self, encoded):
        self.eval()
        self.base.update(force=True)

        gain = self.gain_at(encoded["position"])
        strings = encoded["strings"]

        z_hat = self.base.entropy_bottleneck.decompress(
            strings[1], encoded["shape"]
        )

        scales = self.base.h_s(z_hat)
        indexes = self.base.gaussian_conditional.build_indexes(
            scales
        )

        v_hat = self.base.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )

        y_hat = v_hat / gain
        x_hat = self.base.g_s(y_hat).clamp(0, 1)

        return {"x_hat": x_hat}
