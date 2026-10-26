"""改进 1：文本定位 → ROI 掩膜生成（支持多类别 / 完整句子输入）。

首选 Grounded-SAM2（Grounding DINO + SAM2），轻量备选 CLIPSeg。
详见《改进1-文本定位方法调研.md》。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class MaskResult:
    masks: torch.Tensor                      # [N, H, W] 二值掩膜（阈值 0.5）
    labels: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    probs: Optional[torch.Tensor] = None     # [N, H, W] 软掩膜 0~1，供 TMA 使用


class TextMaskGenerator:
    """统一入口。

    Args:
        method: "grounded_sam2" | "clipseg"
        device: "cuda" | "cpu"
    """

    def __init__(self, method: str = "grounded_sam2", device: str = "cuda", **kwargs):
        self.method = method
        self.device = device
        if method == "grounded_sam2":
            self._impl = GroundedSAM2Generator(device=device, **kwargs)
        elif method == "clipseg":
            self._impl = CLIPSegGenerator(device=device, **kwargs)
        else:
            raise ValueError(f"unknown method: {method}")

    def generate(self, image, prompts: List[str], mode: str = "multiclass") -> MaskResult:
        """image: [3,H,W] float [0,1]；prompts: 文本列表。

        mode:
            "multiclass" —— 点分隔多类别（"sky.clouds.hill"）逐类别出掩膜；
            "sentence"   —— 完整自然语言句（"the dog on the car"）指代式出掩膜。
        """
        return self._impl.generate(image, prompts, mode)


class GroundedSAM2Generator:
    """基于 Grounded-SAM-2（Grounding DINO + SAM2）。

    依赖 scripts/setup clone 的 third_party/Grounded-SAM-2 及其权重。
    具体初始化与调用以该仓库 demo（grounded_sam2_demo）为准，此处给出统一封装骨架。
    """

    def __init__(
        self,
        device: str = "cuda",
        sam2_checkpoint: Optional[str] = None,
        grounding_config: Optional[str] = None,
        grounding_checkpoint: Optional[str] = None,
    ):
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint
        self.grounding_config = grounding_config
        self.grounding_checkpoint = grounding_checkpoint
        # TODO(吴多迅)：按 third_party/Grounded-SAM-2/README.md 初始化
        #  build_sam2 / build GroundingDINO，并加载上述权重后，取消下方注释。
        # from grounding_dino import GroundingDINO  # noqa
        # from sam2.build_sam import build_sam2  # noqa
        raise NotImplementedError(
            "请先完成 Grounded-SAM-2 初始化（见 third_party/Grounded-SAM-2/README.md），"
            "再在此接入 mask 生成逻辑。"
        )

    def generate(self, image, prompts: List[str], mode: str = "multiclass") -> MaskResult:
        # 多类别 -> 点分隔字符串；句子 -> 原句传入 Grounding DINO
        text = ".".join(prompts) if mode == "multiclass" else " ".join(prompts)
        # TODO: 调 Grounding DINO 得到 boxes -> SAM2 得到 masks -> 归一化/上采样
        raise NotImplementedError("接入 Grounded-SAM-2 后实现")


class CLIPSegGenerator:
    """基于 CLIPSeg 的轻量文本→掩膜（可多短语）。

    backend="hf"（默认）走 HuggingFace transformers，权重自动下载/读缓存：
        CIDAS/clipseg-rd64-refined
    backend="local" 走 third_party/clipseg 原始实现，需显式给 weights_path
    （如 rd64-uni.pth），否则输出是随机掩膜。

    输出同时保留二值 masks 与软掩膜 probs，probs 用于对接 TMA。
    """

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "hf",
        model_name: str = "CIDAS/clipseg-rd64-refined",
        weights_path: Optional[str] = None,
        version: str = "ViT-B/16",
        reduce_dim: int = 64,
    ):
        self.device = device
        self.backend = backend

        if backend == "hf":
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            self.processor = CLIPSegProcessor.from_pretrained(model_name)
            self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
            self.model.eval().to(device)
        elif backend == "local":
            try:
                from clipseg.models.clipseg import CLIPDensePredT
            except ImportError as e:
                raise ImportError(
                    "local 后端需要 clipseg：pip install -e third_party/clipseg"
                ) from e
            if weights_path is None:
                raise ValueError(
                    "local 后端必须给 weights_path（如 rd64-uni.pth），"
                    "否则模型是随机初始化，掩膜无意义"
                )
            self.model = CLIPDensePredT(version=version, reduce_dim=reduce_dim)
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(device)
        else:
            raise ValueError(f"unknown backend: {backend}")

    def generate(self, image, prompts: List[str], mode: str = "multiclass") -> MaskResult:
        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise ValueError(f"image 应为 [3,H,W] 或 [1,3,H,W]，实际 {tuple(image.shape)}")

        if self.backend == "hf":
            probs = self._forward_hf(image, prompts)
        else:
            probs = self._forward_local(image, prompts)

        masks = (probs > 0.5).float()
        scores = probs.flatten(1).max(1).values.cpu().tolist()
        return MaskResult(
            masks=masks.cpu(),
            labels=list(prompts),
            scores=scores,
            probs=probs.cpu(),
        )

    def _upsample(self, probs: torch.Tensor, size) -> torch.Tensor:
        """[N,h,w] → [N,H,W]，双线性上采样到原图尺寸。"""
        if probs.dim() == 4 and probs.shape[1] == 1:
            probs = probs[:, 0]
        probs = F.interpolate(
            probs.unsqueeze(1), size=size, mode="bilinear", align_corners=False
        ).squeeze(1)
        return probs

    def _forward_hf(self, image: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        from torchvision.transforms.functional import to_pil_image

        pil = to_pil_image(image.detach().cpu().clamp(0, 1))
        inputs = self.processor(
            text=list(prompts),
            images=[pil] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits       # [N, h, w]
        probs = torch.sigmoid(logits.float())
        return self._upsample(probs, image.shape[-2:])

    def _forward_local(self, image: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            preds = self.model(image.unsqueeze(0).to(self.device), prompts)
        if isinstance(preds, (list, tuple)):
            preds = torch.stack(
                [p.squeeze(0) if p.dim() == 4 else p for p in preds], dim=0
            )  # [N,h,w]
        probs = torch.sigmoid(preds)
        return self._upsample(probs, image.shape[-2:])
