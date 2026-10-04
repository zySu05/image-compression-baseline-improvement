"""改进 1：文本定位 → ROI 掩膜生成（支持多类别 / 完整句子输入）。

首选 Grounded-SAM2（Grounding DINO + SAM2），轻量备选 CLIPSeg。
详见《改进1-文本定位方法调研.md》。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MaskResult:
    masks: torch.Tensor                      # [N, H, W] 二值掩膜
    labels: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


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

    依赖 third_party/clipseg 与 rd64-uni.pth 权重。
    """

    def __init__(
        self,
        device: str = "cuda",
        weights_path: Optional[str] = None,
        version: str = "ViT-B/16",
        reduce_dim: int = 64,
    ):
        self.device = device
        try:
            from clipseg.models.clipseg import CLIPDensePredT
        except ImportError as e:
            raise ImportError("需要 clipseg：先运行 scripts/setup 克隆到 third_party/clipseg") from e
        self.model = CLIPDensePredT(version=version, reduce_dim=reduce_dim)
        if weights_path is not None:
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    def generate(self, image, prompts: List[str], mode: str = "multiclass") -> MaskResult:
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1,3,H,W]
        with torch.no_grad():
            preds = self.model(image.to(self.device), prompts)
        if isinstance(preds, (list, tuple)):
            preds = torch.stack([p if p.dim() == 4 else p.unsqueeze(0) for p in preds], dim=1)
        probs = torch.sigmoid(preds[0])          # [N,H,W]
        masks = (probs > 0.5).float()
        scores = probs.flatten(1).max(1).values.cpu().tolist()
        return MaskResult(masks=masks.cpu(), labels=list(prompts), scores=scores)
