#!/usr/bin/env bash
#
# SoulX-Duplug 本地推理耗时测试（macOS）
#
# 前提：环境已由 install_mac.sh 装好（.venv-mac 虚拟环境 + 模型权重已下载）。
# 依次运行三套基准，并把结果同时打印到屏幕和日志文件。
#
# 用法：
#   bash benchmark_mac.sh                  # 完整基准（三套都跑）
#   bash benchmark_mac.sh --quick          # 快速模式（少量 chunk）
#   bash benchmark_mac.sh --device cpu     # 指定设备（cuda/mps/cpu/auto）
#   bash benchmark_mac.sh --num-chunks 200 # 自定义统计 chunk 数
#   bash benchmark_mac.sh --skip-profile   # 跳过「耗时拆解」
#   bash benchmark_mac.sh --skip-asr       # 跳过「ASR 影响测试」
#   bash benchmark_mac.sh -n myenv         # 自定义虚拟环境目录名
#
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-.venv-mac}"
PYTHON="${REPO_DIR}/${ENV_DIR}/bin/python"
WAV="${REPO_DIR}/assets/tmp.wav"
LORA_CKPT="${REPO_DIR}/pretrained_models/SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth"

DEVICE=""          # 空 = 使用 config.yaml 里的 device（默认 auto）
NUM_CHUNKS=100
QUICK=0
RUN_PROFILE=1
RUN_ASR=1

# ------------------------------------------------------------------ 输出 ------
log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok  ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ fail  ]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
}

# ------------------------------------------------------------------ 参数 ------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)        DEVICE="$2"; shift 2 ;;
    --num-chunks)    NUM_CHUNKS="$2"; shift 2 ;;
    --quick)         QUICK=1; shift ;;
    --skip-profile)  RUN_PROFILE=0; shift ;;
    --skip-asr)      RUN_ASR=0; shift ;;
    -n|--env-dir)    ENV_DIR="$2"; PYTHON="${REPO_DIR}/${ENV_DIR}/bin/python"; shift 2 ;;
    -h|--help)       usage ;;
    *)               die "未知参数: $1（-h 查看用法）" ;;
  esac
done

if [[ ${QUICK} -eq 1 ]]; then
  NUM_CHUNKS=20
fi

# ------------------------------------------------------------------ 日志 ------
TS="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_DIR}/benchmark_results"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/benchmark_${TS}.log"

# ------------------------------------------------------------------ 环境检查 --
echo
log "仓库目录   : ${REPO_DIR}"
log "虚拟环境   : ${ENV_DIR}"
log "日志文件   : ${LOG}"

[[ -x "${PYTHON}" ]] || die "未找到 ${PYTHON}，请先运行: bash install_mac.sh --all"
ok "Python: $("${PYTHON}" -V 2>&1)"

[[ -f "${LORA_CKPT}" ]] || die "未找到模型权重 ${LORA_CKPT}，请先运行: bash install_mac.sh --download-models"
ok "模型权重已就绪"

[[ -f "${WAV}" ]] || die "缺少测试音频 ${WAV}"
ok "测试音频已就绪"

# device 参数处理：只有用户显式指定时才追加 --device
DEVICE_ARG=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARG=(--device "${DEVICE}")
  log "设备       : ${DEVICE}"
else
  log "设备       : 使用 config.yaml 默认值 (auto)"
fi

# 记录开始时间，方便估算总耗时
START_TS=$(date +%s)

# ------------------------------------------------------------------ 工具函数 --
# 跑一个脚本，输出同时进日志；失败不中断，继续跑后面的
run() {
  local name="$1"; shift
  echo
  printf '\033[1;36m==== %s ====\033[0m\n' "$name" | tee -a "${LOG}"
  if "$@" 2>&1 | tee -a "${LOG}"; then
    ok "${name} 完成"
  else
    warn "${name} 失败（已跳过，继续后续测试）"
  fi
}

# ------------------------------------------------------------------ 1. 延迟 --
run "1/3 单 chunk 延迟 + RTF（benchmark_local）" \
  "${PYTHON}" "${REPO_DIR}/benchmark_local.py" \
  --num-chunks "${NUM_CHUNKS}" "${DEVICE_ARG[@]}"

# ------------------------------------------------------------------ 2. 拆解 --
if [[ ${RUN_PROFILE} -eq 1 ]]; then
  run "2/3 耗时逐段拆解（profile_breakdown）" \
    "${PYTHON}" "${REPO_DIR}/profile_breakdown.py" \
    --num-chunks "${NUM_CHUNKS}" "${DEVICE_ARG[@]}"
else
  log "已通过 --skip-profile 跳过耗时拆解"
fi

# ------------------------------------------------------------------ 3. ASR --
if [[ ${RUN_ASR} -eq 1 ]]; then
  if [[ -n "${DEVICE}" ]]; then
    warn "benchmark_asr_impact 不支持 --device，将使用 config.yaml 默认设备"
  fi
  run "3/3 ASR 延迟影响（benchmark_asr_impact）" \
    "${PYTHON}" "${REPO_DIR}/benchmark_asr_impact.py" \
    --num-chunks "${NUM_CHUNKS}"
else
  log "已通过 --skip-asr 跳过 ASR 影响测试"
fi

# ------------------------------------------------------------------ 汇总 ------
END_TS=$(date +%s)
echo
printf '\033[1;36m==== 完成 ====\033[0m\n' | tee -a "${LOG}"
log "总耗时: $(( END_TS - START_TS ))s"
log "完整结果已保存到: ${LOG}"
