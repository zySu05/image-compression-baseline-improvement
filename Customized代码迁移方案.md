# Customized（arXiv:2507.00373）代码阅读笔记与迁移方案

> 负责人：苏致远（组长）
> 任务：读 Customized 官方代码，把 TMA/LMA/CVA 迁移到改进 3 选定的可变码率 baseline。
> 代码位置：`third_party/Customizable-ROI-Based-Deep-Image-Compression/`（由 `scripts/setup` 获取，已 gitignore）

---

## 一、代码已获取

- 仓库：https://github.com/hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression （约 2.1 MB）
- 获取方式：`git clone` 直连被代理重置，改走 codeload zip 解压（`third_party/` 下）
- 关键文件：

| 文件 | 作用 |
|---|---|
| `model_tic.py` | 主模型 `TIC`（990 行），含 LMA/CVA 全部逻辑 |
| `train_tic.py` | 训练 + 损失（含 CVA 的 σ 用法） |
| `eval_kodak.py` | 评测（含 TMA 的文本→掩膜流程） |
| `RSTB.py` / `entropy_model.py` / `transformer_layers.py` | TIC 底座（Swin 块 + 熵模型） |
| `lseg/` | LSeg（lseg-minimal），TMA 的图像侧特征提取 |

## 二、整体架构

- Backbone：**TIC**（Transformer-based Image Compression, DCC 2022）——RSTB（Swin）编码器/解码器 + `CausalAttentionModule` 因果上下文 + 超先验（`EntropyBottleneck` + `GaussianConditional`）。
- 论文的 TMA/LMA/CVA 三个模块，在代码里**没有用这三个名字**，对应关系如下。

## 三、TMA —— 文本掩膜获取（在模型外部，改进 1 的替换点）

- 位置：`eval_kodak.py:184-195`、`train_tic.py:217-237`
- 实现（关键 4 行）：

```python
img_feat  = net_lseg.forward(x_padded)                     # LSeg 图像特征（dense）
prompt    = clip.tokenize(labels).cuda()                   # 文本 → CLIP token
text_feat = net_lseg.clip_pretrained.encode_text(prompt)   # CLIP 文本特征
similarity = cosine_similarity(img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1))  # 逐像素余弦相似度
similarity = similarity.unsqueeze(0)                       # → [1,1,H,W] 文本掩膜
```

- `net_lseg = LSegNet(backbone="clip_vitl16_384", features=256, crop_size=256, ...)`，权重 `lseg_minimal_e200.ckpt`。
- **结论**：TMA = LSeg（图像）+ CLIP（文本）+ 余弦相似度。输入是单个文本标签，输出 `[1,1,H,W]` 相似度图。**这就是改进 1 要替换的对象**——换成 Grounded-SAM2（多类别/句子 → 掩膜），接口保持"文本 → `[1,1,H,W]` 掩膜"即可无缝替换。

## 四、CVA —— σ 权衡因子（硬编码在模型内部）

- 位置：`model_tic.py:403-407`

```python
similarity_loss = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))  # ROI=1.0, 背景=σ=0.01
similarity_imp  = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
similarity_up   = F.interpolate(similarity_loss, scale_factor=2, mode='bilinear')
similarity_up_repeated = similarity_up.repeat(1, 3, 1, 1)
```

- 损失用法（`train_tic.py:113`）：

```python
out["mse_loss"] = torch.mean(((output["x_hat"] - target) * output["similarity"]) ** 2)
```

- **结论**：σ = 0.01 是**背景区域的失真权重**（ROI 权重 1.0，背景误差乘 0.01），通过 `output["similarity"]` 回传到失真损失里。当前 σ 和阈值 0.85 都是**写死的常量**，不是连续条件——这正是"σ 连续化"要改的点。

## 五、LMA —— 隐空间掩膜注意力（隐式比特分配）

- 位置：`model_tic.py:409-424`

```python
similarities_channel = self.simi_net(similarity_imp)       # 掩膜 1ch → M ch（3×stride2 下采样）
similarities_sigmoid = torch.sigmoid(similarities_channel) # 文本侧调制

y_codec    = self.g_a(x, x_size)                           # 编码 → N ch @ H/8
y_codec_a6 = self.g_a6(y_codec)                            # → M ch @ H/16

y_import = self.sub_impor_net(y_codec)                     # 内容侧学习的重要性图
y_tanh   = self.tanh(y_import)
y_soft   = self.softsign(y_tanh)

y_imp = y_soft + similarities_sigmoid                      # 内容重要性 + 文本掩膜调制
y     = y_codec_a6 * y_imp                                 # 隐式比特分配：调制隐变量
```

- `simi_net`（`model_tic.py:271-277`）：`conv(1→64,s2) → ReLU → conv(64→128,s2) → ReLU → conv(128→M,s2)`。
- `sub_impor_net`（`model_tic.py:350-370`）：对 y_codec 做 3 个残差卷积块，输出逐通道重要性图。
- **结论**：LMA = 用 `(学习的内容重要性 + 文本掩膜调制)` 去**乘**隐变量 `y`，ROI 区域得到更大增益 → 更多比特，背景被抑制 → 省比特。这是"隐式比特分配"的核心。

## 六、代码质量问题（迁移时必须注意）

1. **官方评测不是真熵编码**：`eval_kodak.py:218-221` 用 `forward` + `likelihoods` 估算 bpp，不是 `compress`/`decompress` 的真码流。复现指标时要注意口径一致。
2. **compress/decompress 有三个版本且不一致**：`compress`（无 ROI）、`compress_1`/`compress_2`（带 ROI）、`decompress`（需 `flag` 通道裁剪）、`decompress_1`（无 flag）。训练/评测实际走 `forward`，`compress_*` 是半成品。
3. **硬编码**：`device="cuda"`、`N=M=192` 写死在 `decompress`（`model_tic.py:857`）、lseg/数据集/输出路径全是 `/path/to/...`。
4. **通道裁剪 trick**：`compress`/`decompress` 里用 `flag` 记录"全零通道"并在码流里跳过（`model_tic.py:611-630`），迁移到新 baseline 时可简化或丢弃。

## 七、迁移方案（三控合一）

**目标**：在改进 3 选定的可变码率 baseline 上，实现三个可独立调节的控制：

| 控制 | 来源 | 作用 |
|---|---|---|
| `rate`（bpp） | 可变码率 gain（Wang JVCI'25 / Choi gain unit） | 整体码率 |
| `mask`（ROI） | TMA（改进 1：Grounded-SAM2） | 空间/通道级 ROI 调制 |
| `σ`（背景权重） | CVA 连续化 | ROI/背景码率分配权衡 |

核心公式（迁移后的隐变量调制）：

```
y = y_base * gain(rate) * (imp(y_base) + simi(mask, σ))
```

- `gain(rate)`：增益单元，控制整体 bpp（连续）。
- `imp(y_base)`：学习的内容重要性图（沿用 `sub_impor_net`）。
- `simi(mask, σ)`：文本掩膜调制，其中 σ 连续化（见下）。

### 7.1 σ 连续化（CVA 迁移）

把写死的阈值掩膜换成**连续可微**的形式：

```python
# 原文（写死）: w = where(sim > 0.85, 1.0, 0.01)
# 连续化:      w = σ + (1 - σ) * mask_prob        # mask_prob ∈ [0,1]，σ 连续可调
```

σ 作为连续条件信号，同时注入：① 隐变量调制（`simi` 的输入）；② 失真损失权重（`mse = ((x_hat - x) * w)²`）。σ→1 退化为全图等权，σ→0 只保 ROI。

### 7.2 TMA 迁移（对接吴多迅）

Grounded-SAM2 输出多类别/句子的实例掩膜 → 合并成一张 `[1,1,H,W]` 概率图（0~1）→ 直接喂给 `simi_net`。接口不变，替换点在 `eval_kodak.py:184-195`。

### 7.3 落地步骤

1. 跑通官方 `eval_kodak.py`（需 lseg + TIC 权重），校准指标口径（用 forward+likelihood 估算 bpp）。
2. 在可变码率 baseline（TCM 底座）上复刻 `sub_impor_net` + `simi_net` 结构，替换原 TIC 的 RSTB 为 TCM 的注意力块。
3. 加 `gain(rate)` 增益单元，与 `(imp + simi)` 相乘，实现三控。
4. σ 连续化改造损失与调制。
5. 消融：固定 bpp 调 σ、固定 σ 调 bpp，验证三控相互独立。

## 八、风险

- **TIC → TCM 结构差异**：TIC 用 Swin RSTB，可变码率 baseline（TCM 类）用注意力/卷积混合；`sub_impor_net`/`simi_net` 需按新特征尺寸重写。
- **官方代码脏**：compress/decompress 半成品、硬编码多，迁移时应以 `forward` 逻辑为准，重写干净的 compress/decompress。
- **σ 连续化是"权衡"不是"保真提升"**：预期收益按"可控性"写（见《复现阶段任务分工.md》改进 3 定位）。

## 九、对接

- 接**吴多迅**：改进 1 文本定位方法输出（Grounded-SAM2 多类别/句子掩膜）→ 替换 TMA，接口"文本 → `[1,1,H,W]` 掩膜"。
- 接**武力澎**：改进 3 可变码率 baseline（gain 单元）→ 迁移宿主，接口"gain(rate) + 隐变量 y_base"。
