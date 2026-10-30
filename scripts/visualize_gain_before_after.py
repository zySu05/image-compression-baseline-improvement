from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from src.compression.trainable_gain_hyperprior import TrainableGainHyperprior

torch.set_num_threads(4)

# GPU 上沿用已经验证过的 cuDNN 关闭配置，避免此前出现 NaN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEFORE = Path("checkpoints/gain/checkpoint_test.pth")
AFTER = Path("checkpoints/gain/gain_small_trained.pth")
DATA = Path("data/kodak")
OUT = Path("results/visual_comparison")

SAMPLES = [
    "kodim01.png",
    "kodim07.png",
    "kodim13.png",
    "kodim20.png",
]
GAINS = [0.5, 0.75, 1.0, 1.5]

OUT.mkdir(parents=True, exist_ok=True)

def load_model(path):
    ckpt = torch.load(
        path, map_location="cpu", weights_only=True
    )
    model = TrainableGainHyperprior().to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model

print("评价设备:", DEVICE)
before_model = load_model(BEFORE)
after_model = load_model(AFTER)

def to_pil(tensor):
    arr = (
        tensor.detach().cpu()[0]
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(arr)

def reconstruct(model, x, level):
    with torch.no_grad():
        encoded = model.compress(x, level)
        output = model.decompress(encoded)["x_hat"]
        if not torch.isfinite(output).all():
            raise RuntimeError(
                f"重建结果出现 NaN/Inf: level={level}"
            )
        return to_pil(output)

cache = {}

for filename in SAMPLES:
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"样例不存在: {path}")

    with Image.open(path) as im:
        original = im.convert("RGB").copy()
        x = to_tensor(original).unsqueeze(0).to(DEVICE)

    for level, gain in enumerate(GAINS):
        print(f"{filename} | Gain={gain}", flush=True)

        before = reconstruct(before_model, x, level)
        after = reconstruct(after_model, x, level)

        cache[(filename, level)] = (
            original.copy(), before, after
        )

        # 单个样例的训练前后对比图：Gain=1.0
        if level == 2:
            w, h = original.size
            panel = Image.new("RGB", (3 * w, h + 38), "white")
            draw = ImageDraw.Draw(panel)

            for col, (label, img) in enumerate([
                ("Original", original),
                ("Before training", before),
                ("After training", after),
            ]):
                draw.text((col * w + 10, 10), label, fill="black")
                panel.paste(img, (col * w, 38))

            panel.save(
                OUT / f"{Path(filename).stem}_gain_1.00.jpg",
                quality=93,
            )

for level, gain in enumerate(GAINS):
    first = cache[(SAMPLES[0], level)][0]
    w, h = first.size
    row_h = h + 28

    sheet = Image.new(
        "RGB", (4 * w, len(SAMPLES) * row_h), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for row, filename in enumerate(SAMPLES):
        original, before, after = cache[(filename, level)]

        original_np = np.asarray(original).astype(np.int16)
        after_np = np.asarray(after).astype(np.int16)
        diff = Image.fromarray(
            np.clip(np.abs(after_np - np.asarray(before).astype(
                np.int16)) * 4, 0, 255).astype(np.uint8)
        )

        panels = [
            ("Original", original),
            ("Before training", before),
            ("After training", after),
            ("Difference x4", diff),
        ]

        y = row * row_h
        for col, (label, img) in enumerate(panels):
            draw.text(
                (col * w + 8, y + 5),
                f"{filename} | {label}",
                fill="black",
            )
            sheet.paste(img, (col * w, y + 28))

    output = OUT / f"gain_{gain:.2f}_comparison.jpg"
    sheet.save(output, quality=92, optimize=True)
    print("已保存:", output)

print("\n所有对比图生成完成:", OUT)
