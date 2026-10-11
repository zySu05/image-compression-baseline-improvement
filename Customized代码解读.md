# Customized（arXiv:2507.00373）代码解读

> 用途：把官方代码里 TMA / CVA / LMA 三个模块的**逐行解读**整理到项目版本控制内（`third_party/` 被 gitignore，直接写在官方代码上的注释不进 git，故在此留档）。
> 官方代码位置：`third_party/Customizable-ROI-Based-Deep-Image-Compression/`（下文行号均相对该目录）。
> 本文按模块推进：**当前只写了 TMA**，CVA / LMA 后续补充。

---

## 模块索引

| 模块 | 含义 | 本文状态 | 官方代码位置 |
|---|---|---|---|
| **TMA** | 文本掩膜获取 | ✅ 已解读 | `eval_kodak.py` + `model_tic.py` |
| CVA | σ 权衡因子 | ⬜ 待补充 | `model_tic.py:403-407`、`train_tic.py:113` |
| LMA | 隐空间掩膜注意力 | ⬜ 待补充 | `model_tic.py:409-424` |

---

# TMA · 文本掩膜获取（Text Mask Acquisition）

## 1. 它是什么

TMA 是论文三个模块里的**入口**：把用户输入的文本（一个标签词）转成一张空间掩膜 `similarity`，指出"图像里哪些像素是用户关心的 ROI"。

三步：

1. **图像侧**：LSeg（CLIP ViT 视觉塔 + RefineNet）对图像提取逐像素视觉特征 `img_feat`；
2. **文本侧**：CLIP 的 tokenizer + 文本塔把用户标签编码成文本特征 `text_feat`；
3. **融合**：两者按通道 L2 归一化后做**逐像素余弦相似度**，得到 `[1,1,H,W]` 掩膜——越接近 1 越"像"目标文本。

得到的掩膜作为 ROI 信号喂给压缩网络，驱动后续 CVA（失真加权）与 LMA（隐式比特分配）。

## 2. 代码位置

| 文件 | 行号 | 作用 |
|---|---|---|
| `eval_kodak.py` | `main()` 内（约 101-112） | 构建 + 加载 LSeg（评测时使用） |
| `eval_kodak.py` | `inference_clip()`（约 169-200） | **TMA 完整流程**（核心） |
| `model_tic.py` | `__init__()` 内（约 279-290） | 模型内嵌 `net_lseg`（训练时也可用） |

## 3. 核心代码 + 逐行解读

### 3.1 net_lseg 构建（`eval_kodak.py` main 内）

```python
# ======================== TMA · 文本掩膜获取 ========================
# TMA（Text Mask Acquisition）：把用户输入的文本 label 转成一张空间掩膜，
# 指出"哪里是用户关心的 ROI 区域"。这里的 net_lseg 是图像侧特征提取器：
#   LSeg = 用 CLIP ViT-L/16 做 backbone 的"语言驱动"语义分割网络，
#          其 forward 输出的是"逐像素视觉特征"，稍后在 inference_clip 里
#          与 CLIP 文本特征做余弦相似度，得到文本掩膜 similarity。
net_lseg = LSegNet(
        backbone="clip_vitl16_384",   # 图像侧 backbone：CLIP ViT-L/16（输入 384）
        features=256,                 # 输出特征通道数（与 CLIP 对齐的隐维 D）
        crop_size=256,                # 训练裁剪尺寸（评测时图像会被 pad 到 256 的倍数）
        arch_option=0,                # RefineNet 结构选项
        block_depth=0,                # 额外 block 深度
        activation="lrelu",           # 激活函数
    )

# 加载 LSeg 预训练权重（lseg_minimal：官方精简版 checkpoint）
net_lseg.load_state_dict(torch.load("/path/to/lseg/checkpoint/lseg_minimal_e200.ckpt"))
net_lseg.eval()      # 只做特征提取，不求梯度、不更新
net_lseg.to(device)
```

**解读**：`net_lseg` 是 LSeg 模型实例。它内部持有 `clip_pretrained`（CLIP 的 ViT 视觉塔 + 文本塔），这是 TMA 能同时拿到"图像特征"和"文本特征"、并把两者对齐到同一语义空间的关键。

### 3.2 TMA 完整流程（`eval_kodak.py` inference_clip 内）

```python
@torch.no_grad()
def inference_clip(model, x, labels, net_lseg, name):
    """评测时的一次前向：先跑 TMA 得到文本掩膜 similarity，再跑压缩网络。"""

    # ...（图像读取 / pad 到 256 的倍数，略）

    # ---------------- TMA 第 1 步：LSeg 提取图像特征 ----------------
    # net_lseg.forward(x) -> [1, D, H', W'] 逐像素视觉特征（D=features=256，H'/W' 为下采样后的尺寸）
    img_feat = net_lseg.forward(x_padded)
    # 沿通道维 L2 归一化：使后续余弦相似度等价于"单位向量点积"，消除特征模长的影响
    img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)

    # ---------------- TMA 第 2 步：CLIP 提取文本特征 ----------------
    # clip.tokenize：把文本标签转成 token id（自动 pad/截断到 CLIP 的固定长度）
    prompt = clip.tokenize(labels).cuda()
    # 用 LSeg 内部持有的 CLIP 文本塔编码 token -> 文本特征 [1, 512]
    text_feat = net_lseg.clip_pretrained.encode_text(prompt)  # 1, 512
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    # ---------------- TMA 第 3 步：余弦相似度 → 文本掩膜 ----------------
    # 逐像素计算 img_feat 与 text_feat 的余弦相似度：
    #   img_feat_norm:  [1, D, H', W']（每个空间位置是一个 D 维向量）
    #   text_feat_norm: [1, D]（一个 D 维向量），unsqueeze 成 [1, D, 1, 1] 广播到每个像素
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)   # 沿通道维（D）计算
    similarity = cosine_similarity(
        img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
    )   # -> [1, H', W']，每个像素 ∈[-1,1]，越接近 1 越像目标文本
    similarity = similarity.unsqueeze(0)                   # -> [1, 1, H', W'] 作为 ROI 掩膜

    # 之后 model(x_padded, similarity) 里（见 model_tic.py）：
    #   CVA 把它阈值化成 ROI=1/背景=0.01 的失真权重（model_tic.py:403-407）；
    #   LMA 用 simi_net 把它映射成通道调制，乘到隐变量上实现隐式比特分配（model_tic.py:409-424）。
```

**逐行解读**：

- `img_feat = net_lseg.forward(x_padded)` —— LSeg 输出**逐像素**视觉特征（不是分类头），形状 `[1, D, H', W']`。
- `normalize(..., dim=1)` —— 对每个空间位置的 D 维向量做 L2 归一化。归一化后 `cos(u,v) = u·v`，直接点积即可，避免特征模长差异干扰相似度。
- `clip.tokenize(labels)` —— CLIP 的 BPE tokenizer，把文本转成 token id。
- `encode_text(prompt)` —— CLIP 文本塔，输出 `[1, 512]` 的文本向量。
- `CosineSimilarity(dim=1)` —— 沿通道维计算，即"每个像素的 D 维向量"与"文本的 D 维向量"比夹角。
- `text_feat_norm.unsqueeze(-1).unsqueeze(-1)` —— 把 `[1, D]` 变 `[1, D, 1, 1]`，广播到所有空间位置，实现"一次前向对所有像素算相似度"。
- `similarity.unsqueeze(0)` —— 补回 batch 维，得到 `[1, 1, H', W']` 掩膜。

## 4. 关键点 / 易错点

1. **掩膜分辨率不是原图尺寸**：`img_feat` 是 LSeg 下采样后的特征，所以 `similarity` 也是下采样尺寸（`H'×W'`），不是 `H×W`。它进入 `model.forward` 后，`simi_net` 又做 3 次 stride2 下采样，最终对齐到隐变量尺寸。这也是为什么 `eval_kodak.py:179` 要把图像 pad 到 256 的倍数（保证 6 次 stride2 能被整除）。
2. **归一化不可省**：图像/文本特征都要 `normalize`，否则余弦相似度会被模长主导，掩膜质量明显变差。
3. **单一标签输入**：`labels` 是**一个**文本标签（如 `"dog"`）。这正是改进 1 要替换的点——换 Grounded-SAM2 支持**多类别 / 完整句子**，接口保持"文本 → `[1,1,H,W]` 掩膜"即可无缝替换。
4. **与下游衔接**：TMA 输出的 `similarity` 同时被 CVA（阈值化失真权重）和 LMA（`simi_net` 调制）两处消费，是"一次算掩膜、两处用"。

---

# CVA · σ 权衡因子（待补充）

> 位置：`model_tic.py:403-407`、`train_tic.py:113`。后续按上述 TMA 的格式补充。

# LMA · 隐空间掩膜注意力（待补充）

> 位置：`model_tic.py:409-424`、`simi_net`（:271-277）、`sub_impor_net`（:350-370）。后续按上述 TMA 的格式补充。
