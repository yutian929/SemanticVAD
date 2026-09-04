#!/usr/bin/env bash
set -euo pipefail

# 本地客户端环境安装脚本：用 uv 创建 Python 3.10 虚拟环境并安装客户端依赖
# 注意：此环境只用于本地调用（test.py / mic_client.py / record.py），
#       不需要 GPU，也无需安装服务器端那一大堆推理依赖。

cd "$(dirname "$0")"

# 1. 检查 uv 是否可用
if ! command -v uv >/dev/null 2>&1; then
    echo "错误：未找到 uv，请先安装："
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 2. 创建虚拟环境
if [ -d ".venv" ]; then
    echo "虚拟环境 .venv 已存在，跳过创建"
else
    echo "创建虚拟环境 .venv (Python 3.10)..."
    uv venv .venv --python 3.10
fi

# 3. 安装客户端依赖
echo "安装客户端依赖..."
uv pip install --python .venv/bin/python \
    websocket-client \
    numpy \
    soundfile \
    soxr \
    sounddevice

echo ""
echo "安装完成。使用方法："
echo "  source .venv/bin/activate"
echo "  python test.py         # 用 wav 文件测试（需先准备好 assets/tmp.wav）"
