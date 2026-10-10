# Baseline 算法介绍 · 用途与本项目中的作用

> 项目：面向用户意图的语义可控图像压缩方法研究 · 复现改进阶段
> 本文说明本阶段涉及的各 baseline 算法**是什么、干什么用、在本项目里扮演什么角色**。

按四类组织：

1. **学习型图像压缩经典基线**（CompressAI 直调，作 RD 基准）
2. **核心复现/迁移对象**（Customized ROI）
3. **可变码率基线**（改进 3 底座）
4. **文本定位基线**（改进 1 底座）
5. **其余 ROI 参考方法**（按复现策略跳过）

---

## 一、学习型图像压缩经典基线（CompressAI 直调）

> 复现策略：**不额外复现**，直接 `import compressai` 调用预训练模型，仅作为 RD（率失真）基准曲线，用于对比 Customized / 改进方法的性能。

| 算法 | 出处 | 核心思想 | 在本项目中的作用 |
|---|---|---|---|
| **Factorized**（bmshj2018-factorized） | Ballé et al., ICLR 2018 | 端到端自编码器 + 熵瓶颈，无超先验 | 最简熵编码基线，RD 下界参考 |
| **Hyperprior**（bmshj2018-hyperprior） | Ballé et al., ICLR 2018 | 引入**超先验**（边信息 z）估计隐变量分布，进一步去冗余 | 经典超先验基线，Customized 的熵模型（EntropyBottleneck + GaussianConditional）即源于此架构 |
| **Minnen**（mbt2018-mean） | Minnen et al., NeurIPS 2018 | 超先验 + **自回归上下文**（已解码隐变量做均值预测） | 上下文模型基线，代表更强熵模型的上限 |
| **Cheng**（cheng2020-anchor / cheng2020-attn） | Cheng et al., CVPR 2020 | 在超先验基础上加 **attention 模块**，anchor 为无注意力版、attn 为含注意力版 | 性能更强的主流基线；本项目**默认底座用 `cheng2020-anchor`**（configs/default.yaml），也是可变码率 baseline 的常用宿主 |

> 对应 CompressAI zoo 函数名见 [src/compression/variable_rate.py](src/compression/variable_rate.py) 里的 `_MODEL_FNS`。

---

## 二、核心复现/迁移对象：Customized ROI（arXiv:2507.00373）

- **全称**：Customizable ROI-Based Deep Image Compression（TCSVT 2025，Jin et al.）
- **官方代码**：[hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression](https://github.com/hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression)（已 clone 到 `third_party/`，下文路径均相对该目录）

**用途**：在 TIC（Transformer-based Image Compression，DCC 2022）底座上，实现**用户可自定义 ROI（感兴趣区域）**的图像压缩——用户给一段文本，网络据此对文本所指区域分配更多比特、其余背景少分比特。

> 注意：论文里的三个模块 TMA / CVA / LMA 在**代码里没有用这三个名字**，是按功能对应到下列代码片段的（详见 [Customized代码迁移方案.md](Customized代码迁移方案.md)）。

### 2.1 文件地图：算法 ↔ 文件 ↔ 项目功能

| 算法 | 对应文件（相对 `third_party/Customizable-ROI-Based-Deep-Image-Compression/`） | 实现了项目哪项功能 |
|---|---|---|
| **TMA** 文本掩膜获取 | `eval_kodak.py`（`inference_clip`，:184-195）；`model_tic.py`（`net_lseg` 定义，:279-286）；`lseg/` 整个目录 | F1（文本→ROI 定位） |
| **CVA** σ 权衡因子 | `model_tic.py`（:403-407、:445）；`train_tic.py`（损失，:113） | F4（ROI/背景可调权衡，当前写死） |
| **LMA** 隐空间掩膜注意力 | `model_tic.py`（:409-424；`simi_net` :271-277；`sub_impor_net` :350-370；`g_a6` :140） | F3（ROI 隐式比特分配） |
| **TIC 底座 + 熵模型** | `model_tic.py`（`g_a/g_s/h_a/h_s` :305-347；`entropy_bottleneck` :241；`gaussian_conditional` :242；`context_prediction` :243）；`RSTB.py`、`entropy_model.py`、`transformer_layers.py` | F2（压缩编解码） |

数据流一张图：

```
文本 label ──CLIP──▶ text_feat
                         │ 余弦相似度（TMA）
图像 x ──LSeg──▶ img_feat ─┴─▶ similarity 掩膜 [1,1,H,W]
                                      │
        ┌──────────────┬──────────────┤
        │ 阈值化(CVA)  │  simi_net    │
        ▼              ▼              ▼
  失真权重 σ=0.01   similarities_sigmoid（文本调制）
                                     │
  x ─g_a─▶ y_codec ─g_a6─▶ y_codec_a6 · (y_soft + simi) = y （LMA 隐式比特分配）
                                     │
                          h_a→熵编码→h_s→gaussian_conditional→g_s→x_hat
```

---

### 2.2 TMA · 文本掩膜获取（Text Mask Acquisition）

- **是什么**：把用户的文本描述转成一张空间掩膜，指出"哪里是用户关心的区域"。
- **对应文件**：`eval_kodak.py:184-195`（评测时调用）；`lseg/` 目录（LSeg 模型）；`model_tic.py:279-286`（模型内也内嵌了 `net_lseg`）。
- **优点**：
  1. 用 CLIP 的**图文共享语义空间**做零样本定位，**不需要**为 ROI 标注任何分割数据、不需要重训分割头；
  2. 输出是**逐像素余弦相似度**（0~1 软掩膜），连续、可微，能向下游回传梯度；
  3. 与编码网络解耦（模型外计算），换掩膜方法不影响主网络。
- **实现了项目哪个功能**：**F1（文本→ROI 定位）**——语义可控的"入口"。
- **机制（代码解读）**：

```python
# eval_kodak.py:184-195（inference_clip）
img_feat      = net_lseg.forward(x_padded)                       # LSeg 图像侧 dense 特征
img_feat_norm = F.normalize(img_feat, dim=1)
prompt        = clip.tokenize(labels).cuda()                     # 文本 → CLIP token
text_feat     = net_lseg.clip_pretrained.encode_text(prompt)     # CLIP 文本特征 [1,512]
text_feat_norm= F.normalize(text_feat, dim=1)
similarity = CosineSimilarity(dim=1)(                            # 逐像素余弦相似度
    img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1))
similarity = similarity.unsqueeze(0)                             # → [1,1,H,W] 文本掩膜
```

  解读：`img_feat` 与 `text_feat` 都被归一化后做**余弦相似度**，得到每个像素位置"与该文本语义的匹配程度"。匹配度高的地方（文本所指物体）数值接近 1，其余接近 0，即 ROI 掩膜。
  **短板（正是改进 1）**：只输入**单个简单标签**效果好，多类别 / 完整句子（"草地上的人"）支持弱、边界粗糙。

---

### 2.3 CVA · σ 权衡因子（Customizable σ）

- **是什么**：一个控制"ROI 与背景保真优先级"的权重因子 σ。ROI 权重 1.0，背景权重 σ=0.01（写死）。
- **对应文件**：`model_tic.py:403-407`（阈值化 + 上采样）、`:445`（返回 `similarity`）；`train_tic.py:113`（失真损失用法）。
- **优点**：
  1. 通过**失真损失加权**显式表达"ROI 更重要"：背景误差乘 0.01 后几乎不计入损失，网络于是把比特集中到 ROI；
  2. 实现极简（一个 `torch.where`），不引入额外参数。
- **实现了项目哪个功能**：**F4（ROI/背景可调权衡）**——但当前 σ 是**写死的常量**，不是连续可调（这是改进 3 的 (a)）。
- **机制（代码解读）**：

```python
# model_tic.py:403-407
similarity_loss = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
similarity_imp  = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
similarity_up = F.interpolate(similarity_loss, scale_factor=2, mode='bilinear')
similarity_up_repeated = similarity_up.repeat(1, 3, 1, 1)   # → 3 通道，作为失真权重

# train_tic.py:113 —— 加权失真损失
out["mse_loss"] = torch.mean(((output["x_hat"] - target) * output["similarity"]) ** 2)
```

  解读：把掩膜按阈值 0.85 二值化——**ROI 位置权重 = 1.0，背景位置权重 = σ = 0.01**。这个权重图乘到重建误差 `(x_hat - target)` 上再平方求和，等价于"背景的失真被 σ 压到几乎可忽略"，从而引导训练把码率让给 ROI。
  **改造点（改进 3）**：`where(sim>0.85, 1.0, 0.01)` 是离散、写死的，改为连续形式 `w = σ + (1-σ)·mask_prob`，σ 即可连续调节 ROI/背景权衡。

---

### 2.4 LMA · 隐空间掩膜注意力（Latent Mask Attention）

- **是什么**：把 ROI 掩膜**注入隐空间**，用"内容重要性 + 文本调制"去**乘**隐变量，实现对 ROI 多分比特、背景少分比特的**隐式比特分配**。
- **对应文件**：`model_tic.py:409-424`（核心）；`simi_net` :271-277；`sub_impor_net` :350-370；`sub_net_leaky` :254-257；`g_a6` :140。
- **优点**：
  1. **隐式分配**：直接调制隐变量 `y`，ROI 处隐变量增益更大 → 熵编码自然分配更多比特，**无需在码流里显式编码 ROI 位置**；
  2. **双路调制**：`y_soft`（内容重要性，自下而上的 learnable）+ `similarities_sigmoid`（文本掩膜调制，自上而下的语义），两者相加，兼顾内容与语义；
  3. 不改变编解码器主体结构，只在隐变量处插一条调制支路，易迁移。
- **实现了项目哪个功能**：**F3（ROI 隐式比特分配）**——本项目最核心的功能。
- **机制（代码解读）**：

```python
# model_tic.py:409-424（forward 内）
similarities_channel = self.simi_net(similarity_imp)        # 掩膜 1ch → M ch（3×stride2）
similarities_sigmoid = torch.sigmoid(similarities_channel)  # 文本侧调制

y_codec    = self.g_a(x, x_size)   # 编码：x → N ch @ H/8（g_a0..g_a5，3 次 stride2）
y_codec_a6 = self.g_a6(y_codec)    # 再降一次：→ M ch @ H/16

y_import = self.sub_impor_net(y_codec)   # 内容侧重要性图（残差块堆叠）
y_tanh   = self.tanh(y_import)
y_soft   = self.softsign(y_tanh)

y_imp = y_soft + similarities_sigmoid    # 内容重要性 + 文本调制
y     = y_codec_a6 * y_imp               # ★ 隐式比特分配：调制隐变量
```

  两条支路的结构：

```python
# simi_net（model_tic.py:271-277）：掩膜 1ch → M ch，3 次 stride2 下采样
self.simi_net = nn.Sequential(
    conv(1, 64,  k=3, stride=2), nn.ReLU(),
    conv(64, 128, k=3, stride=2), nn.ReLU(),
    conv(128, M, k=3, stride=2),
)

# sub_impor_net（model_tic.py:350-370）：内容重要性图，3 个残差块 + 通道压缩
def sub_impor_net(self, x):
    x1 = self.sub_net_leaky(x)            # stride2 下采样
    x2 = self.sub_net0(x1); x2 = self.sub_net1(x2); x2 = self.sub_net2(x2)
    x2 = x1 + x2                          # 残差 ×3
    x3 = self.sub_net0(x2); ...; x3 = x2 + x3
    x4 = self.sub_net0(x3); ...; x_out = x4 + x3
    x_out = self.sub_net_channel(x_out)   # N → M（1x1）
    return x_out
```

  解读：`y_soft`（内容重要性）与 `similarities_sigmoid`（文本调制）相加后乘回隐变量 `y_codec_a6`。ROI 处掩膜调制值大 → 隐变量被放大 → 量化/熵编码后这些位置信息量更大 → 自然分到更多比特；背景被抑制 → 省比特。这就是"**不显式标 ROI，靠调制隐变量实现比特重分配**"的隐式机制。
  **迁移核心**：整条 LMA 支路要迁到改进 3 的可变码率 baseline 上（见第七节迁移方案）。

---

### 2.5 TIC 底座 + 熵模型（辅助理解）

- **对应文件**：`RSTB.py`（Swin 残差块）、`entropy_model.py`、`transformer_layers.py`；`model_tic.py:305-347`（`g_a/g_s/h_a/h_s`）。
- **作用**：提供编解码 + 熵编码骨架（F2），是 TMA/CVA/LMA 的宿主。
  - `g_a`（编码）/ `g_s`（解码）：RSTB（Swin Transformer 块）+ 卷积交错，多级 stride2 下/上采样；
  - 超先验：`h_a`（超先验编码）→ `EntropyBottleneck` → `h_s`（超先验解码）→ `GaussianConditional`；
  - 上下文模型：`CausalAttentionModule`（`context_prediction`）用已解码的因果上下文预测当前隐变量分布。
- **本项目作用**：TIC 底座本身**不迁移**（改进 3 的基线自带 TCM 底座），迁移时只把 TMA/CVA/LMA 三条支路搬到新底座上。

---

## 三、可变码率基线（改进 3 底座）

> 目标：一个模型实现不同 bpp 的连续调节，作为 Customized 迁移的宿主。调研详见 [改进3-可变码率编码方法调研.md](改进3-可变码率编码方法调研.md)。

| 算法 | 出处 | 机制 | 在本项目中的作用 |
|---|---|---|---|
| **Conditional Autoencoder（gain unit）** | Choi et al., ICCV 2019 | 增益/缩放单元缩放隐变量，实现可变码率 | 可变码率开山之作；`GainUnit` 参考实现已在 [src/compression/variable_rate.py](src/compression/variable_rate.py)，作**短期起步/兜底** |
| **Per-channel gain + 指数插值** | Cui et al. | 逐通道增益 + 指数插值 | 常用连续码率 baseline，作为对比 |
| **Multi-gain + 量化步长 + 线性插值** ⭐ | Wang, Jiang, Sun, Katto, JVCI 2025 | 多增益单元 + 量化步长 + 通道补偿 + 线性插值，无需额外训练/模块即连续 bpp | **推荐 baseline**（BD-rate 1.71%，优于 Cui 2.41%、Tong 7.07%，TCM 底座）——最终迁移宿主 |
| **Multi-Scale Invertible NN** | IEEE TMM 2025 | 多尺度可逆变换 + 扩展增益单元 | 首个单模型超 VVC 的宽码率，作备选对比 |
| **CompressAI 内置** | CompressAI | GainUnit / 可变码率 API | 快速起步，先用多 quality 跑 RD 曲线（见 `scripts/run_variable_rate.py`） |

---

## 四、文本定位基线（改进 1 底座）

> 目标：替换 Customized 的 TMA，支持"多个类别文本 / 完整句子"输入 → ROI 掩膜。调研详见 [改进1-文本定位方法调研.md](改进1-文本定位方法调研.md)。

| 算法 | 出处 | 输入→输出 | 在本项目中的作用 |
|---|---|---|---|
| **LSeg** | Li et al., ICLR 2022 | 多标签文本 → 语义分割 | Customized **现用** TMA；仅支持简单标签，**被替换对象** |
| **CLIPSeg** | Lüddecke & Einhorn, CVPR 2022 | 文本 prompt → 逐像素掩膜 | **轻量备选**，已封装可运行（`CLIPSegGenerator`），适合快速验证 |
| **Grounding DINO** | Liu et al., ECCV 2024 | 自由文本（点分隔类别 + 完整句子）→ 检测框 | 开放集检测，提供文本→框，是 Grounded-SAM2 的检测前端 |
| **SAM2 / Grounded-SAM2** ⭐ | Grounding DINO + SAM2 | 文本 → 实例掩膜 | **首选**：同时支持多类别 + 句子输入，掩膜质量高，替换 TMA 的最终方案 |

---

## 五、其余 ROI 参考方法（按策略跳过）

> 复现策略（见 [复现阶段任务分工.md](复现阶段任务分工.md) 第一节）：**无公开代码/权重的不复现**；有公开代码的只拿来跑测试对比，不深入训练复现。

| 算法 | 出处 | 说明 | 本项目处理 |
|---|---|---|---|
| ROI-based Deep Image Compression with Swin Transformers | arXiv:2305.07783 | Swin 底座 ROI 压缩 | 参考/对比，不深入复现 |
| Super-High-Fidelity via Hierarchical-ROI and Adaptive Quantization | arXiv:2403.13030 | 分层 ROI + 自适应量化 | 参考/对比，不深入复现 |
| ROI-based Deep Image Compression with Implicit Bit Allocation | arXiv:2511.08918 | 隐式比特分配 | **无公开代码，跳过** |

---

## 六、一张图看懂各 baseline 的定位

```
经典压缩基线（Hyperprior/Minnen/Cheng） ──CompressAI直调──▶ RD 基准曲线（只对比，不迁移）

Customized ROI（2507.00373） ──读代码+跑通──▶ 拆出 TMA / CVA / LMA
                                                    │
        TMA ──改进1替换──▶ Grounded-SAM2（多类别/句子 → 掩膜）
                                                    │
        CVA(σ 写死) ──改进3连续化──▶ σ 连续可调权衡
                                                    │
        LMA ──迁移──▶ 迁到 可变码率 baseline（Wang JVCI 2025 多增益）
                                                    │
        三者合体 = 单模型连续调 bpp + 连续调 ROI/背景权重（三控合一）
```
