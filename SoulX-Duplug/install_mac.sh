#!/usr/bin/env bash
#
# SoulX-Duplug macOS (Apple Silicon) 本地推理环境一键安装脚本
#
# 与 install_server.sh 的区别：
#   - 不装 CUDA 版 torch，用 Mac 原生轮子（自带 MPS 后端）
#   - 不装 pynini / WeTextProcessing：openfst 1.8.4 与 pynini 2.1.5/2.1.6 的
#     C++ API 不兼容，arm64 上编译必失败。推理链路只用到 utils/MyTn 里的
#     zh_norm / zh_remove_punc，二者走纯 Python 的 cn_tn.TextNorm，不需要 pynini
#     （textnorm.py 已改为惰性导入）
#   - ASR 默认用 sensevoice（funasr）而非 paraformer：依赖链短，省掉 modelscope
#     连带的 opencv / pandas / oss2 等图像相关依赖
#   - 用 uv 而非 conda 管理环境（Mac 上无需 conda 装 pynini）
#
# 已知性能（M3 / 16GB 实测，仅供参考）：
#   MPS: 平均 288ms/chunk, RTF≈1.80    CPU: 平均 468ms/chunk, RTF≈2.92
#   chunk 实时预算为 160ms，故本机可用于功能验证与离线测试，但达不到实时对话。
#   瓶颈在 ASR：每个 chunk 都要重跑过去 3.2s 音频，单次约 142ms。
#
# 用法：
#   bash install_mac.sh                     # 建 .venv-mac + 装依赖（不下模型）
#   bash install_mac.sh --download-models   # 顺便下载 7.3GB 模型权重
#   bash install_mac.sh --prefetch-asr      # 顺便预下载 ASR 模型
#   bash install_mac.sh --all               # 依赖 + 模型 + ASR 全装
#   bash install_mac.sh -n myenv            # 自定义虚拟环境目录名
#   bash install_mac.sh --skip-brew         # 跳过 ffmpeg/sox 安装
#
set -euo pipefail

# ---------------------------------------------------------------- 基本配置 ----
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="${ENV_DIR:-.venv-mac}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

TORCH_VERSION="2.6.0"
TORCHAUDIO_VERSION="2.6.0"
SETUPTOOLS_VERSION="78.1.1"   # 必须 <81，modelscope 仍在 import pkg_resources
HF_HUB_VERSION="0.34.4"

DO_BREW=1
DO_ENV=1
DO_DEPS=1
DO_VERIFY=1
PREFETCH_ASR=0
DOWNLOAD_MODELS=0

# ------------------------------------------------------------------ 输出 ------
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok   ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ fail  ]\033[0m %s\n' "$*" >&2; exit 1; }

step() {
  echo
  printf '\033[1;36m==== %s ====\033[0m\n' "$*"
}

usage() {
  awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
}

# ------------------------------------------------------------------ 参数 ------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--env-dir)        ENV_DIR="$2"; shift 2 ;;
    --python)            PYTHON_VERSION="$2"; shift 2 ;;
    --skip-brew)         DO_BREW=0; shift ;;
    --skip-env)          DO_ENV=0; shift ;;
    --skip-deps)         DO_DEPS=0; shift ;;
    --skip-verify)       DO_VERIFY=0; shift ;;
    --prefetch-asr)      PREFETCH_ASR=1; shift ;;
    --download-models)   DOWNLOAD_MODELS=1; shift ;;
    --all)               PREFETCH_ASR=1; DOWNLOAD_MODELS=1; shift ;;
    -h|--help)           usage ;;
    *)                   die "未知参数: $1（-h 查看用法）" ;;
  esac
done

cd "${REPO_DIR}"

# ------------------------------------------------------- 0. 平台检查 ---------
step "0/6 平台检查"

[[ "$(uname -s)" == "Darwin" ]] || die "本脚本仅适用于 macOS；Linux/GPU 服务器请用 install_server.sh"

ARCH="$(uname -m)"
log "架构         : ${ARCH}"
log "macOS        : $(sw_vers -productVersion)"
if [[ "${ARCH}" == "arm64" ]]; then
  CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  MEM_GB="$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
  log "芯片         : ${CHIP}"
  log "内存         : ${MEM_GB} GB"
  if (( MEM_GB < 16 )); then
    warn "内存 ${MEM_GB}GB 偏小：fp32 加载模型 + ASR 实测峰值约 7.3GB，可能触发 swap"
  fi
else
  warn "检测到 Intel Mac：没有 MPS 后端，只能跑 CPU，速度会明显更慢"
fi
log "虚拟环境     : ${ENV_DIR} (python ${PYTHON_VERSION})"

# ------------------------------------------------------- 1. 系统依赖 ---------
step "1/6 系统依赖 (ffmpeg / sox)"
if [[ ${DO_BREW} -eq 0 ]]; then
  log "已通过 --skip-brew 跳过"
else
  MISSING=()
  command -v ffmpeg >/dev/null 2>&1 || MISSING+=(ffmpeg)
  command -v sox    >/dev/null 2>&1 || MISSING+=(sox)

  if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "ffmpeg / sox 已就绪"
  elif command -v brew >/dev/null 2>&1; then
    log "brew install ${MISSING[*]}"
    brew install "${MISSING[@]}"
    ok "系统依赖安装完成"
  else
    warn "缺少 ${MISSING[*]}，且未安装 Homebrew。"
    warn "请先装 brew (https://brew.sh) 后重跑，或手动安装这些工具。"
    warn "注意：soundfile/librosa 多数情况下自带解码能力，缺 sox 通常不影响本仓库推理。"
  fi
fi

# ------------------------------------------------------- 2. uv ---------------
step "2/6 uv 包管理器"
if command -v uv >/dev/null 2>&1; then
  ok "uv 已就绪: $(uv --version)"
else
  log "未找到 uv，自动安装..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv 安装脚本默认装到 ~/.local/bin，当前 shell 的 PATH 不会自动刷新
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if command -v uv >/dev/null 2>&1; then
    ok "uv 安装完成: $(uv --version)"
  else
    die "uv 安装后仍不可用，请重启终端后重试，或手动执行：curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
fi

# ------------------------------------------------------- 3. 虚拟环境 ---------
step "3/6 虚拟环境 ${ENV_DIR}"
if [[ ${DO_ENV} -eq 0 ]]; then
  log "已通过 --skip-env 跳过"
elif [[ -x "${ENV_DIR}/bin/python" ]]; then
  ok "${ENV_DIR} 已存在，直接复用: $("${ENV_DIR}/bin/python" -V 2>&1)"
else
  log "uv venv ${ENV_DIR} --python ${PYTHON_VERSION}"
  uv venv "${ENV_DIR}" --python "${PYTHON_VERSION}"
  ok "虚拟环境创建完成"
fi

PY="${REPO_DIR}/${ENV_DIR}/bin/python"
[[ -x "${PY}" ]] || die "找不到 ${PY}，请检查虚拟环境是否创建成功"

# uv pip 统一走这个解释器
upip() { uv pip install --python "${PY}" "$@"; }

# ------------------------------------------------------- 4. Python 依赖 ------
step "4/6 Python 依赖"
if [[ ${DO_DEPS} -eq 0 ]]; then
  log "已通过 --skip-deps 跳过"
else
  # 4.1 torch：Mac 原生轮子自带 MPS，不要加 --index-url 指向 CUDA 源
  log "torch ${TORCH_VERSION} / torchaudio ${TORCHAUDIO_VERSION}（Mac 原生，含 MPS）"
  upip "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

  # 4.2 构建基础。setuptools 必须钉 <81：modelscope/utils/plugins.py 仍 import pkg_resources
  log "构建基础 (setuptools==${SETUPTOOLS_VERSION} / wheel)"
  upip "setuptools==${SETUPTOOLS_VERSION}" wheel

  # 4.3 模型加载与推理
  log "深度学习框架与模型加载"
  upip \
    "numpy==1.26.4" \
    "transformers==4.52.1" \
    "pytorch-lightning==2.5.2" \
    "peft==0.16.0" \
    "accelerate==1.9.0" \
    "safetensors==0.4.5" \
    "sentencepiece==0.2.0" \
    "tokenizers==0.21.4" \
    "omegaconf==2.3.0" \
    "PyYAML==6.0.2" \
    "huggingface_hub==${HF_HUB_VERSION}"

  # 4.4 Web 服务 + 客户端 + 音频 IO + 中文文本处理
  #     这里不装 WeTextProcessing/pynini，理由见文件头注释
  log "Web 服务 / 音频 IO / 中文文本处理"
  upip \
    "fastapi==0.115.3" \
    "uvicorn==0.32.0" \
    "websockets==12.0" \
    "python-multipart==0.0.12" \
    "websocket-client==1.8.0" \
    "soxr==0.5.0.post1" \
    "soundfile==0.12.1" \
    "librosa==0.10.2" \
    "pypinyin==0.55.0" \
    "zhon==2.1.1" \
    "cn2an==0.5.23" \
    "sounddevice"

  # 4.5 级联 ASR：sensevoice 走 funasr，导入链比 modelscope 短很多
  log "级联 ASR (funasr / SenseVoice)"
  upip "funasr==1.2.6"

  # setuptools 可能被上面某些包顶上去，装完回钉一次
  CUR_ST="$("${PY}" -c 'import setuptools;print(setuptools.__version__)' 2>/dev/null || echo 0)"
  if [[ "${CUR_ST}" != "${SETUPTOOLS_VERSION}" ]]; then
    warn "setuptools 被升到 ${CUR_ST}，回退到 ${SETUPTOOLS_VERSION}"
    upip "setuptools==${SETUPTOOLS_VERSION}"
  fi

  ok "Python 依赖安装完成"
fi

# ------------------------------------------------------- 5. 模型 -------------
step "5/6 模型权重"
MODEL_DIR="${REPO_DIR}/pretrained_models"
LORA_CKPT="${MODEL_DIR}/SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth"

if [[ ${DOWNLOAD_MODELS} -eq 1 ]]; then
  # 坑点：如果 shell 里残留 HF_ENDPOINT 指向 hf-mirror，该仓库在镜像上取不到，
  # 会一路报 LocalEntryNotFoundError（看起来像断网，其实不是）。这里显式清掉。
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    warn "检测到 HF_ENDPOINT=${HF_ENDPOINT}，该仓库在镜像上可能不可用，本次下载将忽略它"
  fi
  log "从 HuggingFace 官方源下载 SoulX-Duplug-0.6B 到 pretrained_models/（约 7.3GB）"
  log "该仓库已打包全部三个模型：Qwen3 主干 + glm-4-voice-tokenizer + LoRA 权重"
  env -u HF_ENDPOINT "${PY}" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    "Soul-AILab/SoulX-Duplug-0.6B",
    local_dir="${MODEL_DIR}",
    max_workers=4,
)
print("下载完成:", p)
PY
  ok "权重下载完成"
elif [[ -f "${LORA_CKPT}" ]]; then
  ok "已检测到本地权重: pretrained_models/SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth"
else
  warn "未检测到模型权重，请加 --download-models 重跑，或手动执行："
  warn "  # 注意先 unset HF_ENDPOINT，镜像上没有这个仓库"
  warn "  unset HF_ENDPOINT"
  warn "  ${ENV_DIR}/bin/huggingface-cli download --resume-download \\"
  warn "      Soul-AILab/SoulX-Duplug-0.6B --local-dir pretrained_models"
fi

# ASR 模型默认首次推理时才从 ModelScope 拉取（约 936MB），可提前预热
if [[ ${PREFETCH_ASR} -eq 1 ]]; then
  ASR_NAME="$("${PY}" - <<PY
try:
    from omegaconf import OmegaConf
    print(OmegaConf.load("${REPO_DIR}/config/config.yaml").infer_config.asr.model_name)
except Exception:
    print("sensevoice")
PY
)"
  log "预下载 ASR 模型: ${ASR_NAME}"
  case "${ASR_NAME}" in
    paraformer)
      warn "config.yaml 选的是 paraformer，它依赖 modelscope.pipelines（本脚本未安装其图像依赖）"
      warn "macOS 建议改用 sensevoice；若坚持用 paraformer，请另装 modelscope 相关依赖"
      ;;
    *)
      "${PY}" - <<'PY'
from funasr import AutoModel
# 触发一次加载即完成下载，缓存在 ~/.cache/modelscope
AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=False,
          device="cpu", disable_pbar=True, disable_update=True)
print("SenseVoice 已缓存")
PY
      ;;
  esac
  ok "ASR 模型已缓存到 ~/.cache/modelscope"
else
  log "ASR 模型将在首次推理时自动下载（约 936MB）；想提前拉好可加 --prefetch-asr"
fi

# ------------------------------------------------------- 6. 校验 -------------
step "6/6 安装校验"
if [[ ${DO_VERIFY} -eq 0 ]]; then
  log "已通过 --skip-verify 跳过"
else
  "${PY}" - <<'PY'
import importlib, sys

mods = [
    ("torch", "torch"), ("torchaudio", "torchaudio"),
    ("transformers", "transformers"), ("peft", "peft"),
    ("accelerate", "accelerate"), ("pytorch_lightning", "pytorch-lightning"),
    ("safetensors", "safetensors"), ("sentencepiece", "sentencepiece"),
    ("numpy", "numpy"), ("omegaconf", "omegaconf"), ("yaml", "PyYAML"),
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("websockets", "websockets"),
    ("multipart", "python-multipart"), ("websocket", "websocket-client"),
    ("soxr", "soxr"), ("soundfile", "soundfile"), ("librosa", "librosa"),
    ("pypinyin", "pypinyin"), ("zhon", "zhon"), ("cn2an", "cn2an"),
    ("funasr", "funasr"), ("huggingface_hub", "huggingface_hub"),
    ("setuptools", "setuptools"), ("pkg_resources", "pkg_resources (setuptools<81)"),
]

failed = []
for mod, name in mods:
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {name:<28} {getattr(m, '__version__', '')}")
    except Exception as e:
        failed.append((name, e))
        print(f"  FAIL {name:<28} {type(e).__name__}: {e}")

# 推理链路真正用到的文本正则化函数（不依赖 pynini）
try:
    sys.path.insert(0, ".")
    from utils.MyTn.textnorm import zh_norm, zh_remove_punc
    assert zh_norm("我有2个苹果")
    print("  OK   textnorm (zh_norm/zh_remove_punc)   无需 pynini")
except Exception as e:
    failed.append(("textnorm", e))
    print(f"  FAIL textnorm                     {type(e).__name__}: {e}")

import torch
print()
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  mps available  : {torch.backends.mps.is_available()}")
print(f"  mps built      : {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    x = torch.randn(64, 64, device="mps")
    _ = x @ x
    print("  mps matmul     : OK")
    print("  -> config.yaml 里 device: auto 将解析为 mps")
else:
    print("  警告：MPS 不可用，将退回 CPU（实测约慢 1.6x）")

if failed:
    print(f"\n{len(failed)} 个依赖导入失败，见上方 FAIL 行。")
    sys.exit(1)
PY
  ok "全部依赖导入正常"
fi

# ------------------------------------------------------------------ 收尾 -----
echo
printf '\033[1;32m安装完成。\033[0m\n'
echo
echo "跑延迟基准测试（会先加载模型，首次约 2 分钟）："
echo "  ${ENV_DIR}/bin/python benchmark_local.py --num-chunks 100"
echo "  ${ENV_DIR}/bin/python benchmark_local.py --device cpu     # 对比 CPU"
echo
echo "启动推理服务："
echo "  ${ENV_DIR}/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"
echo
echo "自测（另开一个终端）："
echo "  ${ENV_DIR}/bin/python test.py"
echo
printf '\033[1;33m注意：\033[0m M3/16GB 实测 RTF≈1.80，本机适合功能验证与离线测试，\n'
echo "      实时对话仍建议用 GPU 服务器（config.yaml 的 device: auto 会自动选 cuda）。"
