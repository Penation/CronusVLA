#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-cronusvla}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

if conda env list | awk 'NR > 2 {print $1}' | grep -qx "$ENV_NAME"; then
  echo "[CronusVLA] conda env '$ENV_NAME' already exists; reusing it."
else
  echo "[CronusVLA] creating conda env '$ENV_NAME'..."
  conda create -y -n "$ENV_NAME" python=3.10 pip setuptools wheel
fi

echo "[CronusVLA] installing PyTorch 2.2.0 (CUDA 12.1 wheels)..."
conda run -n "$ENV_NAME" pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[CronusVLA] installing core training dependencies..."
conda run -n "$ENV_NAME" pip install -r "$ROOT_DIR/requirements-train.txt"

echo "[CronusVLA] installing TensorFlow / RLDS dependencies..."
conda run -n "$ENV_NAME" pip install -r "$ROOT_DIR/requirements-train-tf.txt"

echo "[CronusVLA] building flash-attn 2.5.5..."
conda run -n "$ENV_NAME" pip install flash-attn==2.5.5 --no-build-isolation

echo "[CronusVLA] installing conda activation hooks for cuDNN runtime path..."
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"

cat > "$ENV_PREFIX/etc/conda/activate.d/cronusvla.sh" <<'EOF'
if [ -z "${_CRONUSVLA_HOOK_ACTIVE-}" ] && [ -n "${CONDA_PREFIX-}" ]; then
  _CRONUSVLA_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
  _CRONUSVLA_OLD_LD_PRELOAD="${LD_PRELOAD-}"
  _cronusvla_py_ver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  _cronusvla_cudnn_dir="$CONDA_PREFIX/lib/python${_cronusvla_py_ver}/site-packages/nvidia/cudnn/lib"
  if [ -d "$_cronusvla_cudnn_dir" ]; then
    export LD_LIBRARY_PATH="$_cronusvla_cudnn_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -f "$_cronusvla_cudnn_dir/libcudnn_ops_infer.so.8" ]; then
      export LD_PRELOAD="$_cronusvla_cudnn_dir/libcudnn_ops_infer.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
  fi
  export _CRONUSVLA_OLD_LD_LIBRARY_PATH _CRONUSVLA_OLD_LD_PRELOAD _CRONUSVLA_HOOK_ACTIVE=1
  unset _cronusvla_py_ver _cronusvla_cudnn_dir
fi
EOF

cat > "$ENV_PREFIX/etc/conda/deactivate.d/cronusvla.sh" <<'EOF'
if [ -n "${_CRONUSVLA_HOOK_ACTIVE-}" ]; then
  export LD_LIBRARY_PATH="${_CRONUSVLA_OLD_LD_LIBRARY_PATH-}"
  export LD_PRELOAD="${_CRONUSVLA_OLD_LD_PRELOAD-}"
  unset _CRONUSVLA_HOOK_ACTIVE _CRONUSVLA_OLD_LD_LIBRARY_PATH _CRONUSVLA_OLD_LD_PRELOAD
fi
EOF

echo "[CronusVLA] running import checks..."
(
cd "$ROOT_DIR"
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import torch
import tensorflow as tf
import flash_attn
from training.train import TrainConfig

print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_count', torch.cuda.device_count())
print('tensorflow', tf.__version__)
print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))
print('default_vla', TrainConfig(debug=True).vla.vla_id)
PY
)

cat <<EOF
[CronusVLA] environment '$ENV_NAME' is ready.

This host has 4 visible GPUs. The upstream scripts mostly assume 8/16/32/64 GPUs.
For local training on this machine, use:
  torchrun --standalone --nnodes 1 --nproc-per-node 4 training/train.py ...

and override the training config accordingly, e.g.:
  --vla.expected_world_size 4
EOF
