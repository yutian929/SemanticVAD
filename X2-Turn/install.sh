#!/usr/bin/env bash
# One-shot setup for the X2-Turn browser Turn Demo (local Transformers path).
#
# Safe to re-run: every step detects what is already done, and the 10.5 GB
# weight download resumes from partially fetched chunks.
#
# Overrides:
#   ENV_NAME=x2-turn        conda environment name
#   MODEL_DIR=./models/X2-Turn-4B-0812
#   TORCH_SPEC=torch==2.11.0+cu128
#   JOBS=16                 parallel connections for the weight download
#   WEIGHT_SOURCE=auto      auto | modelscope | hf
#   RUN_DEMO=0              1 = launch the browser Turn Demo after setup
#   DEVICE=cuda:0           GPU used when RUN_DEMO=1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-x2-turn}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/X2-Turn-4B-0812}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+cu128}"
JOBS="${JOBS:-16}"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-auto}"
RUN_DEMO="${RUN_DEMO:-0}"
DEVICE="${DEVICE:-cuda:0}"

MS_REPO="x-square-robot/X2-Turn-4B-0812"
MS_API="https://www.modelscope.cn/api/v1/models/$MS_REPO"
PARTS_DIR="$REPO_ROOT/models/.parts"
SMALL_FILES=(config.json generation_config.json params.json processor_config.json
             tekken.json README.md LICENSE NOTICE)

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# A fresh login shell often has conda installed but not initialised (no `conda
# init` in this shell's rc), so `command -v conda` fails even though everything
# is there. Source the first conda.sh we can find before giving up — this makes
# the script re-runnable from any shell without a manual `conda activate`.
if ! command -v conda >/dev/null; then
  # Search the usual prefixes, plus every /home/*/… (the installer may run as
  # root while conda lives under a user's home) and any home matched by the
  # owner of this checkout.
  candidates=("${CONDA_ROOT:-}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"
              /opt/conda /opt/miniforge3 /opt/miniconda3)
  for name in miniforge3 miniconda3 anaconda3; do
    for home in /home/*; do candidates+=("$home/$name"); done
    candidates+=("$(cd "$REPO_ROOT" && stat -c '/home/%U' . 2>/dev/null)/$name")
  done
  for base in "${candidates[@]}"; do
    [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]] || continue
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null || die "conda not found. Install Miniforge or Conda first, or set CONDA_ROOT to its install prefix."
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"

# ---------------------------------------------------------------- 1. conda env
# The yml's pip section always fails: conda rewrites `-e ./pkg[extras]` into a
# temporary requirements file, and pip rejects that form from a requirements
# file. The conda-level packages still land, so tolerate the nonzero exit and
# install the Python deps ourselves in step 2 (the README's documented pip path).
log "Step 1/5  conda environment '$ENV_NAME'"
if [[ -x "$PY" ]]; then
  echo "already exists: $ENV_PREFIX"
else
  conda env create -f environments/environment-transformers.yml || \
    warn "conda reported an error (expected: its pip step cannot handle '-e ./pkg[extras]')"
  [[ -x "$PY" ]] || die "environment was not created at $ENV_PREFIX"
fi
echo "python: $("$PY" --version)"

# ------------------------------------------------------- 2. editable packages
log "Step 2/5  install voxtral-realtime + turn-demo (editable)"
if "$PY" -c 'import demo_turn, voxtral_realtime' 2>/dev/null; then
  echo "already importable, skipping"
else
  "$PY" -m pip install -e "./voxtral-realtime[transformers]" || die "pip install voxtral-realtime failed"
  "$PY" -m pip install -e "./turn-demo"                      || die "pip install turn-demo failed"
fi

# ------------------------------------------------------------------- 3. torch
# PyPI's current torch is a cu130 build that needs a CUDA 13 driver. On a
# CUDA 12.x driver it installs cleanly but silently reports no CUDA, so pin a
# cu128 wheel. transformers>=5.10 only requires torch>=2.5.
log "Step 3/5  verify CUDA-capable torch"
cuda_ok() { "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
if cuda_ok; then
  "$PY" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "-> OK")'
else
  warn "torch cannot see the GPU; installing $TORCH_SPEC"
  "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url http://mirrors.tencentyun.com/pypi/simple \
        --trusted-host mirrors.tencentyun.com "$TORCH_SPEC" || die "torch install failed"
  cuda_ok || die "torch still cannot see the GPU. Check: nvidia-smi, and whether the driver supports this wheel's CUDA version."
  "$PY" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "-> OK")'
fi

# ------------------------------------------------------------------ 4. weights
log "Step 4/5  model weights -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"

if [[ "$WEIGHT_SOURCE" == "auto" ]]; then
  if curl -sS -o /dev/null --max-time 10 https://huggingface.co/api/models/"$MS_REPO" 2>/dev/null; then
    WEIGHT_SOURCE=hf
  else
    warn "huggingface.co unreachable (hf-mirror.com only 308-redirects back to it) -> using ModelScope"
    WEIGHT_SOURCE=modelscope
  fi
fi
echo "source: $WEIGHT_SOURCE"

if [[ "$WEIGHT_SOURCE" == "hf" ]]; then
  "$ENV_PREFIX/bin/huggingface-cli" download "$MS_REPO" --local-dir "$MODEL_DIR" || die "hf download failed"
else
  ms_fetch() {  # $1 = repo-relative path, $2 = destination
    curl -sS -L --max-time 600 --retry 3 -o "$2" "$MS_API/repo?Revision=master&FilePath=$1"
  }

  for f in "${SMALL_FILES[@]}"; do
    if [[ -s "$MODEL_DIR/$f" ]]; then
      echo "  have  $f"
    else
      ms_fetch "$f" "$MODEL_DIR/$f" && echo "  got   $f ($(stat -c%s "$MODEL_DIR/$f") bytes)" \
        || warn "could not fetch $f"
    fi
  done

  TOTAL="$(curl -sS --max-time 60 "$MS_API/repo/files?Revision=master&Recursive=True" \
    | "$PY" -c 'import json,sys; print(next(f["Size"] for f in json.load(sys.stdin)["Data"]["Files"] if f["Path"]=="model.safetensors"))')"
  [[ "$TOTAL" =~ ^[0-9]+$ ]] || die "could not read model.safetensors size from the ModelScope API"
  echo "  model.safetensors: $TOTAL bytes"

  WEIGHTS="$MODEL_DIR/model.safetensors"
  if [[ -f "$WEIGHTS" && "$(stat -c%s "$WEIGHTS")" -eq "$TOTAL" ]]; then
    echo "  already complete, skipping download"
  else
    # ModelScope serves a single stream at only ~550 KB/s (~5 h for 10.5 GB) but
    # honours Range requests, so fetch N chunks concurrently. Each chunk appends
    # to its own part file and asks only for the bytes it is still missing, which
    # makes an interrupted run resume instead of restarting.
    echo "  downloading in $JOBS parallel byte ranges (resumable)"
    mkdir -p "$PARTS_DIR"
    CHUNK=$(( (TOTAL + JOBS - 1) / JOBS ))
    URL="$MS_API/repo?Revision=master&FilePath=model.safetensors"
    for ((i = 0; i < JOBS; i++)); do
      start=$(( i * CHUNK ))
      end=$(( start + CHUNK - 1 ))
      (( end >= TOTAL )) && end=$(( TOTAL - 1 ))
      (
        part="$PARTS_DIR/p$i"; touch "$part"
        need=$(( end - start + 1 ))
        for attempt in $(seq 1 8); do
          have=$(stat -c%s "$part")
          (( have >= need )) && exit 0
          curl -sS -L --max-time 7200 --speed-time 60 --speed-limit 1024 \
               -r "$(( start + have ))-$end" "$URL" >> "$part"
          sleep 3
        done
        have=$(stat -c%s "$part")
        (( have >= need )) || { echo "chunk $i incomplete: $have/$need" >&2; exit 1; }
      ) &
    done
    wait || true

    for ((i = 0; i < JOBS; i++)); do
      [[ -f "$PARTS_DIR/p$i" ]] || die "missing chunk $i — re-run this script to resume"
    done
    : > "$WEIGHTS"
    for ((i = 0; i < JOBS; i++)); do cat "$PARTS_DIR/p$i" >> "$WEIGHTS"; done
    got="$(stat -c%s "$WEIGHTS")"
    if (( got == TOTAL )); then
      rm -rf "$PARTS_DIR"
      echo "  assembled $got bytes -> OK"
    else
      die "size mismatch: got $got, expected $TOTAL. Chunks kept in $PARTS_DIR — re-run to resume."
    fi
  fi
fi

# ------------------------------------------------------------------- 5. verify
log "Step 5/5  verify the checkpoint"
"$PY" - "$MODEL_DIR" <<'PYEOF' || die "checkpoint verification failed"
import json, struct, sys
from pathlib import Path

model_dir = Path(sys.argv[1])
with open(model_dir / "model.safetensors", "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(n))

tensors = [k for k in header if k != "__metadata__"]
expected = 8 + n + max(header[k]["data_offsets"][1] for k in tensors)
actual = (model_dir / "model.safetensors").stat().st_size
assert actual == expected, f"truncated: {actual} != {expected}"
assert "vad_lm_head.weight" in tensors, "turn head missing — wrong checkpoint?"
assert any(k.startswith("base_model.") for k in tensors), "base_model.* keys missing"
print(f"  {len(tensors)} tensors, {actual} bytes, vad_lm_head present -> OK")
PYEOF

printf '\n\033[1;32mSetup complete.\033[0m\n'

if [[ "$RUN_DEMO" == "1" ]]; then
  log "Launching the Turn Demo on $DEVICE (Ctrl-C to stop)"
  cat <<EOF
Open http://localhost:7860, choose '[built-in] English question', then click
'Run scenario'. The first click loads the 4B weights onto the GPU and can take
several minutes. Expected: ASR close to "hello can you tell me what the weather
is like today", ~53 frames of 80 ms, and a turn_end near the end of the utterance.
EOF
  cd "$REPO_ROOT/turn-demo"
  exec env MODEL="$MODEL_DIR" DEVICE="$DEVICE" "$PY" -m demo_turn.server \
    --backend hf --model "$MODEL_DIR" --device "$DEVICE" --host 127.0.0.1 --port 7860
fi

cat <<EOF

Start the demo (or re-run this script with RUN_DEMO=1 to launch it automatically):

  conda activate $ENV_NAME
  cd $REPO_ROOT/turn-demo
  MODEL="$MODEL_DIR" DEVICE=$DEVICE bash run.sh

Wait for 'Uvicorn running on http://127.0.0.1:7860', open http://localhost:7860,
choose '[built-in] English question', then click 'Run scenario'. The first click
loads the 4B weights onto the GPU and can take several minutes.

Expected: ASR close to "hello can you tell me what the weather is like today",
~53 frames of 80 ms, and a turn_end near the end of the utterance.

The checkpoint needs ~24 GB of VRAM. If GPU 0 is busy, use DEVICE=cuda:1.
EOF
