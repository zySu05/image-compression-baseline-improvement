import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.compression.customized_gain_adapter import CustomizedGainAdapter


parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--position", type=float, default=1.5)
parser.add_argument("--sigma", type=float, default=0.3)
args = parser.parse_args()

ckpt = torch.load(
    "checkpoints/gain/gain_small_trained.pth",
    map_location="cpu",
    weights_only=True,
)

model = CustomizedGainAdapter()
model.load_state_dict(ckpt["model"])
model.eval()

with Image.open(args.image) as im:
    x = to_tensor(im.convert("RGB")).unsqueeze(0)

with Image.open(args.mask) as im:
    m = im.convert("L").resize(
        (x.shape[-1], x.shape[-2]),
        Image.Resampling.NEAREST,
    )
    mask = (to_tensor(m) >= 0.5).float().unsqueeze(0)

with torch.no_grad():
    encoded = model.compress_joint(
        x, mask, args.position, args.sigma
    )
    x_hat = model.decompress_joint(encoded)["x_hat"]

assert x_hat.shape == x.shape
assert torch.isfinite(x_hat).all()

bits = sum(
    len(s) * 8
    for group in encoded["strings"]
    for s in group
)

h, w = x.shape[-2:]
latent_bpp = bits / (h * w)
mask_bpp = len(encoded["mask_png"]) * 8 / (h * w)

print("联合编解码成功")
print("输入图像:", args.image)
print("ROI掩膜:", args.mask)
print("Position:", args.position)
print("Sigma:", args.sigma)
print("潜变量bpp:", f"{latent_bpp:.5f}")
print("掩膜bpp:", f"{mask_bpp:.5f}")
print("重建尺寸:", tuple(x_hat.shape))
