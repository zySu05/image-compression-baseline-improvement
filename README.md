# 面向用户意图的语义可控图像压缩方法研究 · 复现改进阶段

> 项目立项（北京交通大学 · 大创）：成果形式 ☑ 软件程序 / 研究报告
> 负责学生：苏致远（24281172）；成员：吴多迅（24281176）、武力澎（24281177）；指导教师：刘美琴

本目录是"复现改进阶段"的工程代码 + 项目说明。三条工作线、复现策略、分工与里程碑见 [复现阶段任务分工.md](复现阶段任务分工.md)。

## 1. 阶段目标

复现 Customizable ROI baseline（arXiv:2507.00373），选定"文本定位"与"可变码率"两个改进底座，形成 2 条可落地的改进方案。

| 成员 | 工作线 | 说明 |
|---|---|---|
| 吴多迅 | 改进 1 文本定位 | 多类别/句子输入 → ROI 掩膜（Grounded-SAM2 主 / CLIPSeg 备） |
| 武力澎 | 改进 3 可变码率 | 单模型多 bpp 编码 baseline |
| 苏致远 | Customized 迁移 | 读官方代码，把 TMA/LMA/CVA 迁移到可变码率 baseline |

## 2. 目录结构

```
├── README.md                   # 项目说明（本文件）
├── requirements.txt            # Python 依赖
├── configs/default.yaml        # 默认配置
├── src/                        # 项目源码
│   ├── evaluation/metrics.py   # 统一评测指标（PSNR/MS-SSIM/LPIPS/ROI-PSNR/bpp）
│   ├── roi/mask_generator.py   # 文本→ROI 掩膜（改进 1）
│   └── compression/variable_rate.py  # 可变码率（改进 3）
├── scripts/                    # 可运行脚本
│   ├── setup.sh / setup.bat    # 环境安装 + clone 三方仓库
│   ├── eval_metrics.py         # 统一评测入口
│   ├── run_text_mask.py        # 改进 1 入口
│   └── run_variable_rate.py    # 改进 3 入口
├── third_party/                # 三方仓库 clone 位置（已 gitignore）
├── data/                       # 数据集（已 gitignore）
├── experiments/                # 实验输出（已 gitignore）
└── *.md                        # 阶段规划 / 调研 / 迁移文档
```

## 3. 环境安装

- Windows：`scripts\setup.bat`
- Linux / macOS / Git Bash：`bash scripts/setup.sh`

脚本做两件事：安装 `requirements.txt` + clone 三方仓库到 `third_party/`。

装完后按各仓库 README 下载预训练权重：

- **Customized** 权重：见其仓库（BaiduDisk / GoogleDrive，提取码 `jooy`）
- **Grounded-SAM-2 / SAM2 / GroundingDINO** 权重：见各仓库 README
- **CLIPSeg** 权重：`rd64-uni.pth` 等

> 说明：Customized 官方代码依赖 `CompressAI 1.2.0b3`，若装最新版出现 API 差异，请按其仓库要求降级。

## 4. 快速开始

### 4.1 统一评测

```bash
python scripts/eval_metrics.py --ref data/kodak/kodim01.png --rec out/recon.png
# 带 ROI 掩膜 + 码流字节数（同时算 ROI-PSNR 和 bpp）
python scripts/eval_metrics.py --ref a.png --rec b.png --mask m.png --bytes 12345
```

### 4.2 改进 1：文本 → ROI 掩膜

```bash
# 多类别（点分隔）
python scripts/run_text_mask.py --image img.jpg --prompt "sky.clouds.hill" --mode multiclass --out out_masks/
# 完整句子（指代式）
python scripts/run_text_mask.py --image img.jpg --prompt "the dog on the car" --mode sentence
# 轻量备选 CLIPSeg
python scripts/run_text_mask.py --image img.jpg --prompt "tree" --method clipseg
```

### 4.3 改进 3：可变码率 RD 曲线

```bash
python scripts/run_variable_rate.py --arch cheng2020-anchor --qualities 1,3,5,6 --images "data/kodak/*.png"
```

> 注意：CompressAI 模型一般要求输入宽高为 64 的倍数（Kodak 768×512 满足）。其他尺寸请先 pad。

## 5. 交付物与里程碑

见 [复现阶段任务分工.md](复现阶段任务分工.md) 第五节。

## 6. 相关文档

- [复现阶段任务分工.md](复现阶段任务分工.md) —— 阶段规划 / 复现策略 / 分工 / 里程碑
- [复现阶段工作汇报框架.md](复现阶段工作汇报框架.md) —— 汇报基稿：功能→算法→主辅→复现/改进 的内在逻辑
- [任务进度记录.md](任务进度记录.md) —— 各模块目前完成情况与待办
- [baseline算法介绍.md](baseline算法介绍.md) —— 各 baseline 用途与在本项目中的作用
- [改进1-文本定位方法调研.md](改进1-文本定位方法调研.md)
- [改进3-可变码率编码方法调研.md](改进3-可变码率编码方法调研.md)
- [Customized代码迁移方案.md](Customized代码迁移方案.md)
