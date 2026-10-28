# 改进3：单模型连续可变码率编码与 ROI 联合压缩

## 1. 工作目标

基于 CompressAI Hyperprior 构建单模型可变码率编码方法，
通过 Gain 调制控制整体压缩码率，并提供 ROI 掩膜与区域权重
参数 sigma 的联合编码接口，为后续 Customized LMA/CVA 模块集成
提供基础。

## 2. 已实现功能

### 2.1 Hyperprior 基线

使用 CompressAI 预训练 Hyperprior Q1—Q4 模型，
完成 Kodak 24 张图像的真实编码、解码和率失真评测。

### 2.2 单模型可变码率

基于一套 Hyperprior Q1 预训练权重，实现潜变量 Gain 调制，
支持不同码率下的真实熵编码与解码。

### 2.3 可学习 Gain

实现四档、192 通道的可学习 Gain 参数，
完成率失真损失计算、梯度验证、DIV2K 小规模训练、
Checkpoint 保存与恢复。

目前仅完成 8 次训练迭代，用于验证训练流程，
不代表模型已经充分收敛或率失真性能达到最优。

### 2.4 连续 Gain 控制

通过相邻档位的对数增益插值，实现非整数 Position 的码率控制。

在 Kodak 24 张图像上测试 9 个插值位置，
完成 216 组真实编解码测试。

本次测试的平均码率范围约为 0.0686—0.1843 bpp。

### 2.5 目标码率搜索

实现 RateController，根据目标 bpp 自动搜索 Gain 插值位置。

在 Kodak 24 张图像上完成 48 组目标码率控制测试。
选定可达目标下的平均相对误差约为 0.1%。

该搜索基于多次真实编码，尚未优化搜索开销。

### 2.6 ROI 与区域权重联合控制

实现接收 ROI 掩膜、Position 和 sigma 的联合编解码接口。

其中：
- Position 控制全局潜变量 Gain；
- ROI Mask 指定需要重点考虑的图像区域；
- sigma 在当前适配原型中表示背景相对权重。

已完成外部 PNG 掩膜输入，以及 CLIPSeg 文本引导掩膜生成
与联合编解码的端到端功能验证。

当前区域调制属于 Hyperprior 简化适配，
并非 Customized 原始 TIC/LMA/CVA 网络的完整复现。

## 3. 主要代码文件

| 文件 | 作用 |
|---|---|
| src/compression/gain_hyperprior.py | 固定 Gain 真实编解码原型 |
| src/compression/trainable_gain.py | 独立可学习增益实验模块 |
| src/compression/trainable_gain_hyperprior.py | 可训练多档 Gain-Hyperprior |
| src/compression/continuous_gain.py | 连续对数 Gain 插值 |
| src/compression/rate_controller.py | 目标 bpp 自动搜索 |
| src/compression/roi_gain_joint.py | Gain 与 ROI 空间调制联合原型 |
| src/compression/customized_gain_adapter.py | ROI/背景权重简化适配 |
| scripts/train_gain_small.py | DIV2K 小规模训练 |
| scripts/evaluate_trained_gain.py | 训练后 Kodak 评测 |
| scripts/evaluate_continuous_gain_kodak.py | 连续码率全量评测 |
| scripts/evaluate_target_bpp_kodak.py | 目标 bpp 控制评测 |
| scripts/test_customized_gain_adapter.py | 外部 ROI 掩膜联合编解码 |

## 4. 联合编解码接口

模型：

```python
from src.compression.customized_gain_adapter import CustomizedGainAdapter

model = CustomizedGainAdapter()
model.load_state_dict(checkpoint["model"])
model.eval()

encoded = model.compress_joint(
    x,
    mask,
    position=1.5,
    sigma=0.3,
)

x_hat = model.decompress_joint(encoded)["x_hat"]
```

输入约定：

- x：RGB 图像张量，形状 [1,3,H,W]，像素归一化到 [0,1]；
- mask：二值 ROI 掩膜，形状 [1,1,H,W]；
- position：连续 Gain 档位位置，当前有效范围 [0,3]；
- sigma：背景相对增益权重，当前适配器要求 0 < sigma <= 1。

当前版本仅验证单张图像的联合编解码。

## 5. 已验证的运行命令

使用与输入图像对应的 ROI PNG 掩膜：

```bash
PYTHONPATH=. python scripts/test_customized_gain_adapter.py \
  --image data/kodak/kodim01.png \
  --mask results/tma_demo/kodim01_clipseg_mask.png \
  --position 1.5 \
  --sigma 0.3
```

上述命令要求本地已经存在：
- gain_small_trained.pth 训练权重；
- Kodak 测试图像；
- 对应的 ROI 掩膜文件。

数据集、预训练模型和实验生成文件不随代码仓库提交，
运行前需在本地准备。

## 6. 与其他模块的集成边界

本部分提供全局 Gain 控制、目标码率搜索、
真实熵编码与解码，以及 ROI 掩膜输入接口。

Customized 相关模块后续可提供：
- TMA 生成的语义 ROI 掩膜；
- LMA/CVA 的正式区域调制方法；
- sigma 的最终控制定义和参数映射。

当前简化区域权重可在后续集成时替换，
但需要同时保证编码端与解码端能够获得一致的控制信息。

## 7. 当前局限

1. Gain 只进行小规模训练，尚未开展充分联合优化。
2. 连续 Gain 插值尚未证明优于多模型率失真基线。
3. 当前 sigma 会影响实际总 bpp，尚未证明总码率与区域质量完全独立控制。
4. ROI 掩膜及控制元数据存在额外传输开销。
5. 正式 Customized TIC/LMA/CVA 网络尚未完整移植。
6. 本阶段重点是功能跑通，不代表完整论文复现。

## 8. 下一步

与团队现有 TMA/LMA/CVA 模块完成接口对接，
在同一模型中实现总码率和 ROI/背景权重的联合控制，
随后开展必要的联合训练和性能验证。

## 9. 与 Customized LMA/CVA 模块的接口约定

### 9.1 当前已提供能力

当前 Gain-Hyperprior 模型支持：

- 连续 Position 调节；
- 目标 bpp 搜索；
- 外部 ROI 掩膜输入；
- sigma 区域权重调制；
- 编码端与解码端一致的调制参数恢复；
- 真实熵编码、解码和图像重建。

### 9.2 联合调用接口

```python
encoded = model.compress_joint(
    x,
    mask,
    position=1.5,
    sigma=0.3,
)
x_hat = model.decompress_joint(encoded)["x_hat"]
```

输入要求：

- x 为 [1,3,H,W] 的 RGB 图像；
- mask 为 [1,1,H,W] 的 ROI 掩膜；
- position 位于 [0,3]；
- 当前适配器的 sigma 位于 (0,1]。

### 9.3 Customized 模块接入位置

当前区域调制入口为：

`spatial_gain(mask, y_shape, position, sigma)`

后续可根据正式 LMA/CVA 模块输出，
替换或扩展区域调制方式。

若正式 LMA/CVA 依赖 TIC 编码器内部特征，
需要进一步修改编码器和解码器接口，
不能直接视为简单的函数替换。

### 9.4 对接注意事项

- 编码端和解码端必须使用一致的区域调制信息；
- 若引入可学习模块，应同步加载其训练参数；
- ROI 掩膜及控制参数需要考虑传输开销；
- 当前版本是简化适配原型，不等于原始 Customized 完整复现；
- 后续需验证总码率与 ROI/背景质量控制的联合效果。
