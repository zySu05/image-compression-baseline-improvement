import csv
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.continuous_gain import ContinuousGainHyperprior
from src.compression.rate_controller import RateController

torch.set_num_threads(4)

ckpt = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True,
)

model = ContinuousGainHyperprior()
model.load_state_dict(ckpt["model"])
model.eval()

controller = RateController(model, max_steps=8)

images = sorted(Path("data/kodak").glob("kodim*.png"))
assert len(images) == 24

rows = []

with torch.no_grad():
    for index, path in enumerate(images, 1):
        with Image.open(path) as im:
            x = to_tensor(im.convert("RGB")).unsqueeze(0)

        pixels = x.shape[-2] * x.shape[-1]

        def get_bpp(position):
            encoded = model.compress_at(x, position)
            bits = sum(
                len(s) * 8
                for group in encoded["strings"]
                for s in group
            )
            return bits / pixels

        min_bpp = get_bpp(0)
        max_bpp = get_bpp(3)

        for fraction in [0.25, 0.75]:
            target = (
                min_bpp + fraction * (max_bpp - min_bpp)
            )

            result = controller.compress_to_target(x, target)
            x_hat = model.decompress_at(
                result["encoded"]
            )["x_hat"]

            mse = torch.mean((x - x_hat) ** 2).item()
            psnr = (
                -10 * math.log10(mse)
                if mse > 0 else float("inf")
            )

            rows.append({
                "image": path.name,
                "target_fraction": fraction,
                "target_bpp": target,
                "actual_bpp": result["actual_bpp"],
                "abs_error": result["error"],
                "position": result["position"],
                "psnr_db": psnr,
                "search_encodes": result["search_encodes"],
            })

        print(f"[{index:02d}/24] {path.name} 完成", flush=True)

output = Path("results/continuous/target_bpp_kodak.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n测试完成，记录数:", len(rows))
print("结果保存至:", output)
