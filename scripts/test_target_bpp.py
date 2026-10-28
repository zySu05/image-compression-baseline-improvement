import math
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.continuous_gain import (
    ContinuousGainHyperprior
)

torch.set_num_threads(4)

ckpt = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True
)

model = ContinuousGainHyperprior()
model.load_state_dict(ckpt["model"])
model.eval()

with Image.open("data/kodak/kodim01.png") as im:
    x = to_tensor(im.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def measure(position):
    encoded = model.compress_at(x, position)

    bits = sum(
        len(s) * 8
        for group in encoded["strings"]
        for s in group
    )

    bpp = bits / (x.shape[-2] * x.shape[-1])
    return bpp, encoded


@torch.no_grad()
def search_target_bpp(target, max_steps=10):
    left, right = 0.0, 3.0

    low_bpp, low_encoded = measure(left)
    high_bpp, high_encoded = measure(right)

    if not low_bpp <= target <= high_bpp:
        raise ValueError(
            f"目标 {target:.5f} 超出当前图像可达范围 "
            f"[{low_bpp:.5f}, {high_bpp:.5f}]"
        )

    best = min(
        [
            (abs(low_bpp - target), left, low_bpp, low_encoded),
            (abs(high_bpp - target), right, high_bpp, high_encoded)
        ],
        key=lambda item: item[0]
    )

    for _ in range(max_steps):
        mid = (left + right) / 2
        bpp, encoded = measure(mid)

        candidate = (
            abs(bpp - target), mid, bpp, encoded
        )

        if candidate[0] < best[0]:
            best = candidate

        if bpp < target:
            left = mid
        else:
            right = mid

    _, position, actual_bpp, encoded = best

    x_hat = model.decompress_at(encoded)["x_hat"]
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = -10 * math.log10(mse) if mse > 0 else float("inf")

    return position, actual_bpp, psnr


for target in [0.12, 0.15, 0.20]:
    position, bpp, psnr = search_target_bpp(target)

    print(
        f"目标={target:.3f} | "
        f"position={position:.5f} | "
        f"实际bpp={bpp:.5f} | "
        f"误差={abs(bpp-target):.5f} | "
        f"PSNR={psnr:.3f} dB",
        flush=True
    )
