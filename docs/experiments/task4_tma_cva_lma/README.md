# 任务四：TMA → CVA → LMA 图像压缩展示结果

## 一、实验说明

输入图像：Kodak kodim01
文本提示词：building
掩膜阈值：0.95
运行模式：真实比特流编码与解码

## 二、前后图片

- kodim01_before.png：压缩前原图
- kodim01_after.png：压缩后重建图

## 三、实际运行命令

在项目根目录、已配置好 Python 环境及本地模型资源的 WSL Ubuntu 环境执行：

```bash
PYTHONPATH=/tmp/roi-paper-upstream:$PWD \
python -u scripts/run_paper_tma_cva_fast.py \
  --image data/kodak/kodim01.png \
  --prompt "building" \
  --mode bitstream \
  --clipseg-backend hf \
  --clipseg-weights checkpoints/clipseg \
  --threshold 0.95 \
  --out-dir results/paper_tma_cva_fast/building_kodim01_t095
```

## 四、运行结果

- ROI 覆盖率：13.47%
- PSNR：16.5782 dB
- 实际比特率：0.060547 BPP
- 输出尺寸：(1, 3, 512, 768)

以上指标来自已完成的实际运行记录。

## 五、复现说明

命令中的 /tmp/roi-paper-upstream 是本地上游代码目录。
运行前需要准备相应的代码、TIC 权重、CLIPSeg 本地权重和输入图像。
这些本地资源不因上传本目录而自动包含在 GitHub 中。
