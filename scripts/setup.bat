@echo off
REM 环境安装 + 三方仓库 clone（Windows）
REM 用法: scripts\setup.bat
cd /d "%~dp0\.."

echo [1/3] 安装 Python 依赖...
python -m pip install -U pip
python -m pip install -r requirements.txt

echo [2/3] clone 三方仓库到 third_party\ ...
if not exist third_party mkdir third_party

if not exist third_party\Customizable-ROI-Based-Deep-Image-Compression (
  git clone --depth 1 https://github.com/hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression.git third_party\Customizable-ROI-Based-Deep-Image-Compression
) else (
  echo   skip Customized 已存在
)

if not exist third_party\Grounded-SAM-2 (
  git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git third_party\Grounded-SAM-2
) else (
  echo   skip Grounded-SAM-2 已存在
)

if not exist third_party\clipseg (
  git clone --depth 1 https://github.com/timojl/clipseg.git third_party\clipseg
) else (
  echo   skip clipseg 已存在
)

echo [3/3] 完成。下一步：按各仓库 README 下载预训练权重。
