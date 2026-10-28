import math

import torch
import torch.nn.functional as F

from src.compression.roi_gain_joint import ROIGainJoint


class CustomizedGainAdapter(ROIGainJoint):
    """
    Customized-inspired ROI weight adapter.

    position: controls global Gain.
    sigma: relative background weight, 0 < sigma <= 1.

    This is a functional Hyperprior adaptation,
    not the original pretrained TIC/CVA model.
    """

    def spatial_gain(self, mask, y_shape, position, sigma):
        sigma = float(sigma)

        if not math.isfinite(sigma) or not 0 < sigma <= 1:
            raise ValueError("sigma must be in (0, 1]")

        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [B,1,H,W]")

        m = F.interpolate(
            mask.float(),
            size=y_shape[-2:],
            mode="nearest",
        )

        m = (m >= 0.5).to(
            device=self.log_gain.device,
            dtype=self.log_gain.dtype,
        )

        global_gain = self.gain_at(position)

        # ROI weight = 1, background weight = sigma
        region_weight = m + (1.0 - m) * sigma

        return global_gain * region_weight
