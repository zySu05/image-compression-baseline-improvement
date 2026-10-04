# Customized（arXiv:2507.00373）代码迁移方案

> 负责人：苏致远（组长）
> 任务：读 Customized 官方代码，把 TMA/LMA/CVA 迁移到改进 3 选定的可变码率 baseline。

---

## 一、代码信息

- 官方仓库：https://github.com/hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression
- 底座：TIC，依赖 CompressAI 1.2.0b3；文本 ROI 定位用 **LSeg**（lseg-minimal）。
- 权重：BaiduDisk / GoogleDrive 提供预训练权重（提取码 `jooy`）。
- 训练：`python train_tic.py -d /path/to/dataset/COCO/ --cuda`
- 评测：`eval_kodak.py`（默认支持文本输入 ROI 定位）。

## 二、要读透的三个模块

| 模块 | 作用 | 迁移要点 |
|---|---|---|
| TMA | 文本 → ROI 掩膜（现用 LSeg/CLIP） | 替换为改进 1 选定的 Grounding DINO + SAM2，接口对齐 |
| LMA | 隐空间掩膜注意力（ROI 隐式调制） | 找到 mask 注入隐变量的位置，迁移到可变码率 baseline |
| CVA | σ 权衡因子（ROI/背景码率分配） | σ 离散档 → 连续条件信号，与可变码率 gain 协同 |

## 三、迁移步骤

1. 跑通官方 eval（Kodak），校准指标（与原文误差 < 0.3 dB）。
2. 读 TMA/LMA/CVA 代码，画出"掩膜生成 → 隐变量调制 → 损失加权"的数据流。
3. 确定 mask 注入点与 σ 加权点，标注在可变码率 baseline（TCM）对应位置。
4. 迁移：在可变码率 baseline 上实现"gain（bpp） + mask 注意力（ROI） + σ（背景权重）"三控。
5. 消融验证：固定 bpp 调 σ、固定 σ 调 bpp，证明三控相互独立可调。

## 四、风险

- TIC → TCM 结构差异：mask 注意力模块需按 TCM 特征尺寸适配。
- LSeg 依赖较重：迁移后先用改进 1 的 Grounding DINO 输出替换。

## 五、与另外两人的对接

- 接**吴多迅**：改进 1 选定的文本定位方法输出 → 替换 TMA，接口为"文本（多类/句子）→ 归一化 mask"。
- 接**武力澎**：改进 3 选定的可变码率 baseline → 作为迁移宿主，接口为"gain（bpp 控制）+ 隐变量"。
