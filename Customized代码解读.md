# Customized（arXiv:2507.00373）代码解读

> 用途：把官方代码里 TMA / CVA / LMA 三个模块的**逐行解读**整理到项目版本控制内（`third_party/` 被 gitignore，直接写在官方代码上的注释不进 git，故在此留档）。
> 官方代码位置：`third_party/Customizable-ROI-Based-Deep-Image-Compression/`（下文行号均相对该目录）。
> 本文按模块推进：**当前只写了 TMA**，CVA / LMA 后续补充。

---

## 模块索引

| 模块 | 含义 | 本文状态 | 官方代码位置 |
|---|---|---|---|
| **TMA** | 文本掩膜获取 | ✅ 已解读 | `eval_kodak.py` + `model_tic.py` |
| **CVA** | σ 权衡因子 | ✅ 已解读 | `model_tic.py:403-407`、`train_tic.py:113` |
| **LMA** | 隐空间掩膜注意力 | ✅ 已解读 | `model_tic.py:409-424`、`simi_net`、`sub_impor_net` |

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

# CVA · σ 权衡因子（Customizable σ）

## 1. 它是什么

CVA 是论文的"**可定制权衡因子**"：用一个标量 **σ** 显式控制 ROI 与背景的**失真优先级**。ROI 权重 = 1.0，背景权重 = σ（官方**写死 σ = 0.01**）。

做法：把 TMA 的软掩膜阈值化成一张"ROI=1 / 背景=σ"的权重图，乘到重建误差上再求 MSE。这样**背景的误差被 σ 压到几乎忽略不计**，训练时网络自然把比特集中到 ROI、背景少分比特。

一句话：**CVA 通过"失真损失加权"告诉网络——ROI 的保真更重要，背景可以糊。**

## 2. 代码位置

| 文件 | 行号 | 作用 |
|---|---|---|
| `model_tic.py` | `forward()` 内（:403-407） | 阈值化 + 上采样 + 重复到 3 通道 |
| `model_tic.py` | `forward()` 返回（:445） | 把权重图放进 `output["similarity"]` |
| `train_tic.py` | `RateDistortionLoss.forward`（:113） | 加权 MSE 失真 |

## 3. 核心代码 + 逐行解读

### 3.1 权重图生成（`model_tic.py` forward 内）

```python
# CVA：把 TMA 的软掩膜 similarity 阈值化成"ROI=1 / 背景=σ"的二值权重图
#   similarity > 0.85  → 1.0   （ROI：保真优先）
#   否则               → 0.01  （背景：σ=0.01，失真几乎不计）
similarity_loss = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
similarity_imp  = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))

# 上采样 2 倍（双线性）：LSeg 输出 stride=2（H/2×W/2），×2 后对齐到重建图 H×W
similarity_up = F.interpolate(similarity_loss, scale_factor=2, mode='bilinear')
# 从 [1,1,H,W] repeat 成 [1,3,H,W]，匹配 RGB 3 通道，才能和 (x_hat - target) 逐元素相乘
similarity_up_repeated = similarity_up.repeat(1, 3, 1, 1)
```

返回值里带上它（`model_tic.py:445`）：

```python
return {
    "y_hat": y_hat,
    "y": y,
    "similarity": similarity_up_repeated,   # 交给损失函数做加权
    "x_hat": x_hat,
    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
}
```

### 3.2 加权失真（`train_tic.py` RateDistortionLoss）

```python
# 逐像素加权 MSE：背景误差 ×0.01，ROI 误差 ×1.0，再平方平均
out["mse_loss"] = torch.mean(((output["x_hat"] - target) * output["similarity"]) ** 2)
# 总损失 = λ·255²·加权MSE + 码率（λ 为拉格朗日系数，控制 RD 权衡）
out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
```

**逐行解读**：

- `torch.where(similarity > 0.85, 1.0, 0.01)` —— 把连续相似度按阈值 **0.85** 二值化：>0.85 的位置判为 ROI（权重 1.0），其余判为背景（权重 **σ=0.01**）。
- `similarity_loss` 与 `similarity_imp` **内容完全相同**，只是两个用途：前者给失真损失加权（CVA），后者喂给 `simi_net` 做隐空间调制（LMA）。两处的阈值 0.85 和 σ=0.01 都是**写死的常量**。
- `scale_factor=2` —— 上采样 2 倍对齐重建图尺寸。LSeg 输出是 stride=2 的 `H/2×W/2`，×2 后正好是 `H×W`，与 `x_hat`/`target` 对齐。
- `repeat(1, 3, 1, 1)` —— 单通道权重图复制成 3 通道，与 RGB 重建图逐像素相乘。
- 加权 MSE —— `(x_hat - target) * similarity` 里，背景像素的误差被乘 0.01 后平方，几乎等于 0；ROI 像素误差保留。等价于"只在乎 ROI 的保真"。

## 4. 关键点 / 易错点

1. **σ 是"权衡"不是"保真提升"**：σ=0.01 意味着背景失真被大幅忽略，ROI/背景是**此消彼长**的关系，不是"背景也变好"。这正是学长指出的逻辑点——改进 3 的定位是"连续可调权衡（可控性）"，不是"背景保真提升"。
2. **σ 和阈值都是写死的离散常量**：`0.85`、`0.01` 硬编码，不是一个可调条件。**这是改进 3 的 (a) 要改的点**——把 `where(sim>0.85, 1.0, 0.01)` 换成连续形式 `w = σ + (1-σ)·mask_prob`，σ 即可连续调节 ROI/背景权衡。
3. **一张权重图两处用**：`similarity_loss`（→ 失真损失）和 `similarity_imp`（→ LMA 的 `simi_net`）本质是同一张二值图。迁移时若做 σ 连续化，要保证这两处注入的是**同一个连续 σ 信号**，才能让"损失"与"调制"一致。
4. **分辨率对齐依赖 LSeg 的 stride=2**：这里 `scale_factor=2` 是硬编码，隐含假设"LSeg 输出 = H/2"。换掉 TMA（改进 1 换 Grounded-SAM2）后，掩膜分辨率可能不同，上采样倍数要相应调整，不能照抄 2×。

---

# LMA · 隐空间掩膜注意力（Latent Mask Attention）

## 1. 它是什么

LMA 是论文**最核心**的机制：把 ROI 掩膜**注入隐空间**，用「内容重要性 + 文本掩膜调制」去**乘**隐变量，实现对 ROI 多分比特、背景少分比特的**隐式比特分配**。

核心公式：

```
y = y_codec_a6 * (y_soft + similarities_sigmoid)
                 └─────────── y_imp ───────────┘
    编码隐变量     内容重要性      文本掩膜调制
                 IG（重要性图）   MR（掩膜表示）
```

LMA 由论文里的**两个子模块**组成，`y_imp` 就是两者的融合：

| 子模块 | 论文含义 | 提取的先验 | 对应代码 |
|---|---|---|---|
| **MR**（Mask Representation，掩膜表示） | 从文本掩膜提取"哪里是 ROI" | latent spatial prior（空间先验） | `simi_net` |
| **IG**（Importance map Generation，重要性图生成） | 从图像内容生成"哪些细节对重建更重要" | latent RDO prior（率失真先验） | `sub_impor_net` |

两者相加得到 latent attention map（`y_imp`），再乘到隐变量 `y_codec_a6` 上。IG 的作用是**避免隐变量从头学习**（复用内容本身的重要性），MR 提供**语义层面的空间引导**。

**为什么叫"隐式"**：不显式在码流里标出 ROI 位置，而是**调制隐变量的幅度**——ROI 处调制值大 → 隐变量被放大 → 量化/熵编码后这些位置信息量更大 → 自然分到更多比特；背景被抑制 → 省比特。比特分配完全由隐变量自身幅度决定，无需额外 ROI 标记。

## 2. 代码位置

| 文件 | 行号 | 作用 |
|---|---|---|
| `model_tic.py` | `forward()` 内（:409-424） | **LMA 核心**：调制隐变量 |
| `model_tic.py` | `simi_net`（:271-277） | **MR（掩膜表示）**：掩膜 1ch → M ch |
| `model_tic.py` | `sub_impor_net`（:350-370） | **IG（重要性图生成）**：内容侧重要性图 |
| `model_tic.py` | `sub_net_leaky`/`sub_net0/1/2`/`sub_net_channel`（:254-269） | sub_impor_net 的子结构 |
| `model_tic.py` | `g_a6`（:140）、`g_a`（:305-315） | 编码器（产生 `y_codec` / `y_codec_a6`） |

## 3. 核心代码 + 逐行解读

### 3.1 两条调制支路的结构

**MR · 文本侧 `simi_net`（`model_tic.py:271-277`）**：把 1 通道掩膜映射成 M 通道调制（提取隐空间空间先验），3 次 stride2 下采样：

```python
self.simi_net = nn.Sequential(
    conv(1, 64,  kernel_size=3, stride=2),   # 1→64，H/2→H/4
    nn.ReLU(),
    conv(64, 128, kernel_size=3, stride=2),  # 64→128，H/4→H/8
    nn.ReLU(),
    conv(128, M, kernel_size=3, stride=2),   # 128→M，H/8→H/16
)
```

**IG · 内容侧 `sub_impor_net`（`model_tic.py:350-370`）**：对编码特征学一张"哪里重要"的图（生成隐空间 RDO 先验），先 stride2 下采样再堆 3 个残差块：

```python
def sub_impor_net(self, x):   # important map（输入 y_codec @ H/8）
    x1 = self.sub_net_leaky(x)          # conv(N,N,s2)+LeakyReLU：H/8 → H/16
    x2 = self.sub_net0(x1); x2 = self.sub_net1(x2); x2 = self.sub_net2(x2)
    x2 = x1 + x2                        # 残差块 ①
    x3 = self.sub_net0(x2); x3 = self.sub_net1(x3); x3 = self.sub_net2(x3)
    x3 = x2 + x3                        # 残差块 ②
    x4 = self.sub_net0(x3); x4 = self.sub_net1(x4); x4 = self.sub_net2(x4)
    x_out = x4 + x3                     # 残差块 ③
    x_out = self.sub_net_channel(x_out) # conv(N,M,1x1)：通道压缩 N→M
    return x_out
```

其中子结构（`model_tic.py:254-269`）：

```python
self.sub_net_leaky = nn.Sequential(conv(N, N, k=3, s=2), nn.LeakyReLU())  # 下采样
self.sub_net0      = nn.Sequential(conv(N, 64, k=1, s=1), nn.ReLU())       # 1x1 降维
self.sub_net1      = nn.Sequential(conv(64, 64, k=3, s=1), nn.ReLU())      # 3x3 特征
self.sub_net2      = conv(64, N, k=1, s=1)                                  # 1x1 升维（残差出口）
self.sub_net_channel = conv(N, M, k=1, s=1)                                 # N→M 通道对齐
```

### 3.2 隐式比特分配（`model_tic.py` forward 内）

```python
# ---- MR（掩膜表示）：文本侧 → 隐空间空间先验 ----
similarities_channel = self.simi_net(similarity_imp)        # 掩膜 [1,1,H/2,W/2] → [1,M,H/16,W/16]
similarities_sigmoid = torch.sigmoid(similarities_channel)  # sigmoid → (0,1)

# ---- 编码 ----
y_codec    = self.g_a(x, x_size)   # 编码：x → N ch @ H/8（g_a0..g_a5，共 3 次 stride2）
y_codec_a6 = self.g_a6(y_codec)    # 再降一次：→ M ch @ H/16

# ---- IG（重要性图生成）：内容侧 → 隐空间 RDO 先验 ----
y_import = self.sub_impor_net(y_codec)   # y_codec @ H/8 → 重要性图 @ H/16
y_tanh   = self.tanh(y_import)           # tanh 压缩到 (-1,1)
y_soft   = self.softsign(y_tanh)         # softsign 再压缩到 (-1,1)（更平滑）

# ---- MR + IG 融合 → 隐式比特分配 ----
y_imp = y_soft + similarities_sigmoid     # IG(重要性图) + MR(掩膜表示) → [1,M,H/16,W/16]
y     = y_codec_a6 * y_imp               # ★ 调制隐变量：ROI 放大、背景抑制

# 之后 y 进入超先验 + 熵编码（h_a → entropy_bottleneck → h_s → gaussian_conditional）
```

**逐行解读**：

- `similarities_sigmoid`（**MR**）：掩膜经 `simi_net` 后 `sigmoid` 到 (0,1)，是**自上而下**的语义调制（隐空间空间先验）——文本所指区域值大、其余值小。
- `y_soft`（**IG**）：`y_codec` 经 `sub_impor_net` + `tanh` + `softsign`，是**自下而上**的、可学习的内容重要性图（隐空间 RDO 先验），范围 (-1,1)。
- `y_imp = y_soft + similarities_sigmoid`：MR 与 IG 相加，兼顾"内容本身重不重要"（IG）与"是不是用户关心的语义区域"（MR）。
- `y = y_codec_a6 * y_imp`：乘回隐变量。ROI 处 `y_imp` 大 → 隐变量幅度大 → 熵编码分配更多比特；背景被抑制 → 省比特。**这就是"隐式比特分配"的核心**。
- 激活函数分工：`sigmoid`（文本，正调制 0~1）+ `softsign(tanh())`（内容，可正可负 -1~1），允许内容支路既能增强也能抑制。

## 4. 关键点 / 易错点

1. **三路都要汇聚到 H/16**：`simi_net` 3 次 stride2（掩膜 H/2 → H/16）、`g_a6` 1 次 stride2（y_codec H/8 → H/16）、`sub_net_leaky` 1 次 stride2（y_codec H/8 → H/16）。**下采样次数不是随便设的**，是由"输入分辨率 → 隐变量分辨率"的差距决定的。这也是为什么 `simi_net` 恰好 3 个 stride2（因为掩膜在 H/2，而不是原图 H）。
2. **两条支路都要把通道数对齐到 M**：`simi_net` 末层 `conv(128→M)`、`sub_net_channel` `conv(N→M)`，否则 `y_soft + similarities_sigmoid` 通道对不上。
3. **这是迁移的核心**：整条 LMA（`simi_net` + `sub_impor_net` + 乘法调制）要搬到改进 3 的可变码率 baseline 上。但新 baseline（TCM 类底座）的隐变量尺寸/通道数与 TIC 不同，`sub_impor_net`/`simi_net` 的 stride 次数和通道数都要按新特征重写（对应 [Customized代码迁移方案.md](Customized代码迁移方案.md) 第七节）。
4. **官方 `compress`/`decompress` 是半成品**：LMA 的隐式分配逻辑只存在于 `forward` 里；`compress`/`decompress` 版本不一致且未完整实现，训练/评测实际都走 `forward`（见迁移方案第六节）。迁移时应以 `forward` 为准，重写干净的压缩/解压。
5. **与 CVA 共用同一张掩膜**：`similarity_imp`（给 LMA）和 `similarity_loss`（给 CVA）是同一张二值图。迁移做 σ 连续化时，要保证两处注入的是同一个连续 σ 信号。
