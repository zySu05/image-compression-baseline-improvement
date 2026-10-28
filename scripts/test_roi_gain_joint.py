import math

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.roi_gain_joint import ROIGainJoint

torch.set_num_threads(4)

ckpt = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True,
)

model = ROIGainJoint()
model.load_state_dict(ckpt["model"])
model.eval()

with Image.open("data/kodak/kodim01.png") as im:
    x = to_tensor(im.convert("RGB")).unsqueeze(0)

_, _, h, w = x.shape
mask = torch.zeros(1, 1, h, w)
mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1

roi = mask.expand_as(x)
bg = 1 - roi

for position in [1.0, 2.0]:
    for sigma in [0.0, 0.2, 0.4]:
        encoded = model.compress_joint(
            x, mask, position, sigma
        )
        x_hat = model.decompress_joint(encoded)["x_hat"]

        assert x_hat.shape == x.shape
        assert torch.isfinite(x_hat).all()

        bits = sum(
            len(s) * 8
            for group in encoded["strings"]
            for s in group
        )
        latent_bpp = bits / (h * w)
        mask_bpp = len(encoded["mask_png"]) * 8 / (h * w)

        error = (x - x_hat).square()
        roi_mse = (error * roi).sum() / roi.sum()
        bg_mse = (error * bg).sum() / bg.sum()

        roi_psnr = -10 * math.log10(max(roi_mse.item(), 1e-12))
        bg_psnr = -10 * math.log10(max(bg_mse.item(), 1e-12))

        print(
            f"position={position:.1f} "
            f"sigma={sigma:.1f} "
            f"latent_bpp={latent_bpp:.5f} "
            f"mask_bpp={mask_bpp:.5f} "
            f"ROI_PSNR={roi_psnr:.3f} "
            f"BG_PSNR={bg_psnr:.3f}",
            flush=True
        )

print("ROI + Gain联合编解码测试完成")
