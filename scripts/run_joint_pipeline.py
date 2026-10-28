import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from transformers import (
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
)

from src.compression.customized_gain_adapter import CustomizedGainAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--position", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--clipseg-dir",
        default="checkpoints/clipseg"
    )
    parser.add_argument(
        "--gain-ckpt",
        default="checkpoints/gain/gain_small_trained.pth"
    )
    parser.add_argument("--out", default="results/joint_demo")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image) as im:
        image = im.convert("RGB")
    x = to_tensor(image).unsqueeze(0)

    print("[1/3] 加载CLIPSeg并生成ROI掩膜", flush=True)
    processor = CLIPSegProcessor.from_pretrained(
        args.clipseg_dir, local_files_only=True
    )
    segmenter = CLIPSegForImageSegmentation.from_pretrained(
        args.clipseg_dir, local_files_only=True
    ).eval()

    inputs = processor(
        text=[args.prompt],
        images=[image],
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = segmenter(**inputs).logits

    if logits.ndim == 2:
        logits = logits[None, None]
    elif logits.ndim == 3:
        logits = logits[:, None]

    prob = F.interpolate(
        torch.sigmoid(logits),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    )

    mask = (prob >= args.threshold).float()
    to_pil_image(mask[0]).save(out / "roi_mask.png")

    # CLIPSeg完成后释放内存
    del segmenter, processor

    print("[2/3] 加载Gain并执行联合编解码", flush=True)
    checkpoint = torch.load(
        args.gain_ckpt,
        map_location="cpu",
        weights_only=True
    )

    model = CustomizedGainAdapter()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        encoded = model.compress_joint(
            x, mask, args.position, args.sigma
        )
        x_hat = model.decompress_joint(encoded)["x_hat"]

    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()

    to_pil_image(x_hat[0]).save(out / "reconstructed.png")

    bits = sum(
        len(stream) * 8
        for group in encoded["strings"]
        for stream in group
    )
    pixels = image.width * image.height

    latent_bpp = bits / pixels
    mask_bpp = len(encoded["mask_png"]) * 8 / pixels

    print("[3/3] 完成", flush=True)
    print("输入图像:", args.image)
    print("文本描述:", args.prompt)
    print("ROI占比:", round(mask.mean().item(), 4))
    print("Position:", args.position)
    print("Sigma:", args.sigma)
    print("潜变量bpp:", round(latent_bpp, 5))
    print("掩膜bpp:", round(mask_bpp, 5))
    print("重建结果:", out / "reconstructed.png")
    print("ROI掩膜:", out / "roi_mask.png")
    print("联合流程运行成功")


if __name__ == "__main__":
    main()
