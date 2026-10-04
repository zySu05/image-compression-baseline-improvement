#!/usr/bin/env bash
# 环境安装 + 三方仓库 clone（Linux / macOS / Git Bash）
# 用法: bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/3] 安装 Python 依赖..."
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "[2/3] clone 三方仓库到 third_party/ ..."
mkdir -p third_party
clone() {  # $1=url  $2=目录名
  if [ -d "third_party/$2/.git" ]; then
    echo "  skip $2（已存在）"
  else
    git clone --depth 1 "$1" "third_party/$2"
  fi
}
clone https://github.com/hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression.git Customizable-ROI-Based-Deep-Image-Compression
clone https://github.com/IDEA-Research/Grounded-SAM-2.git Grounded-SAM-2
clone https://github.com/timojl/clipseg.git clipseg

echo "[3/3] 完成。下一步：按各仓库 README 下载预训练权重。"
