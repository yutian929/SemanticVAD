#!/usr/bin/env bash
#
# SoulX-Duplug 推理服务端一键安装脚本
#
# 严格按照 requirements-inference.txt 头部约定的顺序安装：
#   1) conda install -c conda-forge pynini=2.1.5      # WeTextProcessing 的依赖，pip 装需编译 OpenFst，极易失败
#   2) pip install torch==2.6.0 torchaudio==2.6.0 \
#        --index-url https://download.pytorch.org/whl/cu124 \
#        --extra-index-url https://mirrors.aliyun.com/pypi/simple/
#   3) pip install -r requirements-inference.txt -i https://mirrors.aliyun.com/pypi/simple/
#
# torch / torchaudio 不在 requirements-inference.txt 中，由本脚本第 2 步按 CUDA 版本单独安装。
#
# 用法：
#   bash install_server.sh                       # 默认：创建 conda 环境 soulx-duplug (py3.10) + cu124 + 阿里云镜像
#   bash install_server.sh --cuda cu121          # 换 CUDA 轮子（cu118 / cu121 / cu124 / cpu）
#   bash install_server.sh --no-mirror           # 用官方 PyPI，不走阿里云镜像
#   bash install_server.sh -n myenv --python 3.10
#   bash install_server.sh --skip-env            # 装到当前已激活的环境里，不新建
#   bash install_server.sh --prefetch-asr        # 顺便把 config.yaml 里选中的 ASR 模型预下载好
#   bash install_server.sh --download-models     # 顺便从 HuggingFace 拉 SoulX-Duplug 权重到 pretrained_models/
#
set -euo pipefail

# ---------------------------------------------------------------- 基本配置 ----
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REPO_DIR}/requirements-inference.txt"

ENV_NAME="${ENV_NAME:-soulx-duplug}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu124}"

TORCH_VERSION="2.6.0"
TORCHAUDIO_VERSION="2.6.0"
PYNINI_VERSION="2.1.5"
SETUPTOOLS_VERSION="78.1.1"   # 必须 <81，modelscope 仍在 import pkg_resources

MIRROR_URL="https://mirrors.aliyun.com/pypi/simple/"
MIRROR_HOST="mirrors.aliyun.com"
USE_MIRROR=1

DO_APT=1
DO_ENV=1
DO_PYNINI=1
DO_TORCH=1
DO_REQS=1
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
  # 打印文件开头的注释块（跳过 shebang，遇到第一行非注释即停）
  awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
}

# ------------------------------------------------------------------ 参数 ------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--env-name)      ENV_NAME="$2"; shift 2 ;;
    --python)           PYTHON_VERSION="$2"; shift 2 ;;
    --cuda)             CUDA_TAG="$2"; shift 2 ;;
    --no-mirror)        USE_MIRROR=0; shift ;;
    --skip-apt)         DO_APT=0; shift ;;
    --skip-env)         DO_ENV=0; shift ;;
    --skip-pynini)      DO_PYNINI=0; shift ;;
    --skip-torch)       DO_TORCH=0; shift ;;
    --skip-reqs)        DO_REQS=0; shift ;;
    --skip-verify)      DO_VERIFY=0; shift ;;
    --prefetch-asr)     PREFETCH_ASR=1; shift ;;
    --download-models)  DOWNLOAD_MODELS=1; shift ;;
    -h|--help)          usage ;;
    *)                  die "未知参数: $1（-h 查看用法）" ;;
  esac
done

[[ -f "${REQ_FILE}" ]] || die "找不到 ${REQ_FILE}"

case "${CUDA_TAG}" in
  cu118|cu121|cu124|cpu) ;;
  *) die "--cuda 只支持 cu118 / cu121 / cu124 / cpu，收到: ${CUDA_TAG}" ;;
esac

# pip 通用参数
PIP_ARGS=(--disable-pip-version-check)
if [[ ${USE_MIRROR} -eq 1 ]]; then
  PIP_ARGS+=(-i "${MIRROR_URL}" --trusted-host "${MIRROR_HOST}")
fi

log "仓库目录     : ${REPO_DIR}"
log "conda 环境   : ${ENV_NAME} (python ${PYTHON_VERSION})"
log "torch 轮子   : ${TORCH_VERSION}+${CUDA_TAG}"
log "PyPI 源      : $([[ ${USE_MIRROR} -eq 1 ]] && echo "${MIRROR_URL}" || echo '官方 PyPI')"

# ------------------------------------------------------- 0. 系统依赖 ----------
step "0/6 系统依赖 (ffmpeg / sox / libsox-dev)"
if [[ ${DO_APT} -eq 0 ]]; then
  log "已通过 --skip-apt 跳过"
else
  MISSING=()
  command -v ffmpeg >/dev/null 2>&1 || MISSING+=(ffmpeg)
  command -v sox    >/dev/null 2>&1 || MISSING+=(sox)
  if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "ffmpeg / sox 已就绪"
  elif command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
      command -v sudo >/dev/null 2>&1 && SUDO="sudo" \
        || warn "非 root 且无 sudo，跳过系统包安装；请自行安装: ${MISSING[*]}"
    fi
    if [[ "$(id -u)" -eq 0 || -n "${SUDO}" ]]; then
      log "安装: ffmpeg sox libsox-dev"
      ${SUDO} apt-get update -qq
      ${SUDO} apt-get install -y ffmpeg sox libsox-dev
      ok "系统依赖安装完成"
    fi
  else
    warn "缺少 ${MISSING[*]}，且本机没有 apt-get；请用系统包管理器手动安装 ffmpeg sox libsox-dev"
  fi
fi

# ------------------------------------------------------- conda 定位 ----------
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  for cand in "${CONDA_EXE:-}" \
              "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
              "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" \
              "/opt/conda/bin/conda" "/usr/local/miniconda3/bin/conda"; do
    [[ -n "${cand}" && -x "${cand}" ]] && { CONDA_BIN="${cand}"; break; }
  done
fi

if [[ -n "${CONDA_BIN}" ]]; then
  CONDA_BASE="$("${CONDA_BIN}" info --base)"
  PS1="${PS1:-}"   # 部分 conda 版本的 activate 脚本在 set -u 下会因 PS1 未定义而报错
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  ok "找到 conda: ${CONDA_BIN}"
else
  warn "未找到 conda。pynini 只能靠 pip 编译安装，成功率很低。"
  warn "建议先装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  [[ ${DO_ENV} -eq 1 ]] && die "无 conda 无法创建环境；已有 venv 的话请加 --skip-env 重跑"
fi

# ------------------------------------------------------- 1. conda 环境 -------
step "1/6 conda 环境"
if [[ ${DO_ENV} -eq 0 ]]; then
  log "已通过 --skip-env 跳过，安装到当前环境: $(command -v python)"
else
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    ok "环境 ${ENV_NAME} 已存在，直接复用"
  else
    log "创建环境 ${ENV_NAME} (python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"
  ok "已激活: $(python -V 2>&1) @ $(command -v python)"
fi

PY="$(command -v python)"
[[ -n "${PY}" ]] || die "当前环境没有 python"

# ------------------------------------------------------- 2. pynini -----------
step "2/6 pynini=${PYNINI_VERSION} (conda-forge)"
if [[ ${DO_PYNINI} -eq 0 ]]; then
  log "已通过 --skip-pynini 跳过"
elif "${PY}" -c "import pynini" >/dev/null 2>&1; then
  ok "pynini 已安装: $("${PY}" -c 'import pynini;print(pynini.__version__)' 2>/dev/null || echo unknown)"
elif [[ -n "${CONDA_BIN}" ]]; then
  log "conda install -c conda-forge pynini=${PYNINI_VERSION}"
  conda install -y -c conda-forge "pynini=${PYNINI_VERSION}" \
    || die "pynini 安装失败。它是 WeTextProcessing 的硬依赖，pip 装需要现场编译 OpenFst，请优先解决 conda 通道问题后重试。"
  ok "pynini 安装完成"
else
  die "没有 conda，无法安装 pynini；请安装 Miniconda 后重跑，或自行编译 OpenFst 再加 --skip-pynini"
fi

# setuptools 必须先钉到 <81：conda 默认给 84.x，而 modelscope/utils/plugins.py 仍 import pkg_resources
log "钉住 setuptools==${SETUPTOOLS_VERSION}（modelscope 需要 pkg_resources）"
"${PY}" -m pip install "${PIP_ARGS[@]}" "setuptools==${SETUPTOOLS_VERSION}" wheel

# ------------------------------------------------------- 3. torch ------------
step "3/6 torch ${TORCH_VERSION} / torchaudio ${TORCHAUDIO_VERSION} (${CUDA_TAG})"
if [[ ${DO_TORCH} -eq 0 ]]; then
  log "已通过 --skip-torch 跳过"
else
  TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
  TORCH_ARGS=(--index-url "${TORCH_INDEX}")
  if [[ ${USE_MIRROR} -eq 1 ]]; then
    TORCH_ARGS+=(--extra-index-url "${MIRROR_URL}" --trusted-host "${MIRROR_HOST}")
  fi
  log "pip install torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_INDEX}"
  "${PY}" -m pip install --disable-pip-version-check \
    "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" "${TORCH_ARGS[@]}"
  ok "torch 安装完成"
fi

# ------------------------------------------------------- 4. requirements -----
step "4/6 requirements-inference.txt"
if [[ ${DO_REQS} -eq 0 ]]; then
  log "已通过 --skip-reqs 跳过"
else
  log "pip install -r $(basename "${REQ_FILE}")"
  "${PY}" -m pip install "${PIP_ARGS[@]}" -r "${REQ_FILE}"
  # 某些包（datasets / funasr 等）会顺手把 setuptools 升上去，装完再钉一次
  CUR_ST="$("${PY}" -c 'import setuptools;print(setuptools.__version__)' 2>/dev/null || echo 0)"
  if [[ "${CUR_ST}" != "${SETUPTOOLS_VERSION}" ]]; then
    warn "setuptools 被升到 ${CUR_ST}，回退到 ${SETUPTOOLS_VERSION}"
    "${PY}" -m pip install "${PIP_ARGS[@]}" "setuptools==${SETUPTOOLS_VERSION}"
  fi
  ok "Python 依赖安装完成"
fi

# ------------------------------------------------------- 5. 模型 -------------
step "5/6 模型权重"
MODEL_DIR="${REPO_DIR}/pretrained_models"
if [[ ${DOWNLOAD_MODELS} -eq 1 ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  log "从 HuggingFace 下载 SoulX-Duplug-0.6B 到 ${MODEL_DIR}（HF_ENDPOINT=${HF_ENDPOINT}）"
  "${PY}" -m pip install "${PIP_ARGS[@]}" -U "huggingface_hub[cli]"
  "${PY}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Soul-AILab/SoulX-Duplug-0.6B", local_dir="${MODEL_DIR}")
PY
  ok "权重下载完成"
elif [[ -f "${MODEL_DIR}/SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth" ]]; then
  ok "已检测到本地权重: pretrained_models/SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth"
else
  warn "未检测到 ${MODEL_DIR}/SoulX-Duplug/*.pth"
  warn "请加 --download-models 重跑，或手动执行:"
  warn "  export HF_ENDPOINT=https://hf-mirror.com"
  warn "  huggingface-cli download --resume-download Soul-AILab/SoulX-Duplug-0.6B --local-dir pretrained_models"
fi

# ASR 模型（paraformer / sensevoice）默认在首次请求时才从 ModelScope 拉取，可提前预热
if [[ ${PREFETCH_ASR} -eq 1 ]]; then
  ASR_NAME="$("${PY}" - <<PY
try:
    from omegaconf import OmegaConf
    print(OmegaConf.load("${REPO_DIR}/config/config.yaml").infer_config.asr.model_name)
except Exception:
    print("paraformer")
PY
)"
  log "预下载 ASR 模型: ${ASR_NAME}"
  case "${ASR_NAME}" in
    sensevoice)
      "${PY}" - <<'PY'
from modelscope import snapshot_download
snapshot_download("iic/SenseVoiceSmall")
PY
      ;;
    *)
      "${PY}" - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    revision="v2.0.4",
)
PY
      ;;
  esac
  ok "ASR 模型已缓存到 ~/.cache/modelscope"
else
  log "ASR 模型（config.yaml 的 infer_config.asr.model_name）将在首次推理时自动下载；"
  log "想提前拉好可加 --prefetch-asr"
fi

# ------------------------------------------------------- 6. 校验 -------------
step "6/6 安装校验"
if [[ ${DO_VERIFY} -eq 0 ]]; then
  log "已通过 --skip-verify 跳过"
else
  "${PY}" - <<'PY'
import importlib, sys

# (import 名, 展示名)
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
    ("pynini", "pynini"), ("tn", "WeTextProcessing"),
    ("funasr", "funasr"), ("modelscope", "modelscope"),
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

import torch
print()
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  cuda build     : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  gpu            : {torch.cuda.get_device_name(0)}")
    print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
else:
    print("  警告：未检测到可用 GPU。config.yaml 里 device: cuda / precision: bf16，")
    print("       CPU 上无法直接跑通，请检查驱动或换匹配的 --cuda 轮子。")

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
echo "启动推理服务："
if [[ ${DO_ENV} -eq 1 ]]; then
  echo "  conda activate ${ENV_NAME}"
fi
echo "  cd ${REPO_DIR}"
echo "  bash run.sh          # uvicorn server:app --host 127.0.0.1 --port 8000"
echo
echo "自测（另开一个终端，同一环境下）："
echo "  python example_client.py"
