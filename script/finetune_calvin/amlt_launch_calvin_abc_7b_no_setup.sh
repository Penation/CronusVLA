#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_DIR}/script/finetune_calvin/finetune_7b_calvin_abc_1node8gpu.sh}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:?DATA_ROOT_DIR is required}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:?RUN_ROOT_DIR is required}"
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-cronusvla}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/scratch/cronusvla_models}"
BASE_LLM_REPO="${BASE_LLM_REPO:-NousResearch/Llama-2-7b-hf}"
CRONUS_REPO_ID="${CRONUS_REPO_ID:-JeasLee/cronusvla_7B_bridge_rt_1}"
CRONUS_CKPT_NAME="${CRONUS_CKPT_NAME:-step-055000-epoch-04-loss=0.0286.pt}"

DATA_VERSION_DIR="${DATA_ROOT_DIR}/custom_finetuning/1.0.0"
EXPECTED_DATA_FILES="${EXPECTED_DATA_FILES:-1158}"
DATA_WAIT_SECONDS="${DATA_WAIT_SECONDS:-60}"
DATA_WAIT_MAX_POLLS="${DATA_WAIT_MAX_POLLS:-360}"

export PATH="${CONDA_ROOT}/bin:${PATH}"

echo "[stage 1/4][$(date -Iseconds)] Validate dataset and runtime"
pwd
nvidia-smi
df -h /scratch /mnt/cronusvla_blob 2>/dev/null || true
test -f "${RUN_SCRIPT}"
test -f "${REPO_DIR}/training/train.py"
mkdir -p "${RUN_ROOT_DIR}" "${LOCAL_MODEL_ROOT}" "${HF_HOME}" "${TORCH_HOME}"

current_files=0
for ((poll_idx=1; poll_idx<=DATA_WAIT_MAX_POLLS; poll_idx++)); do
  current_files="$(find "${DATA_VERSION_DIR}" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -f "${DATA_VERSION_DIR}/dataset_info.json" && "${current_files}" -ge "${EXPECTED_DATA_FILES}" ]]; then
    echo "dataset_root=${DATA_ROOT_DIR}"
    echo "dataset_files=${current_files}/${EXPECTED_DATA_FILES}"
    break
  fi
  echo "waiting_for_dataset poll=${poll_idx} files=${current_files}/${EXPECTED_DATA_FILES}"
  sleep "${DATA_WAIT_SECONDS}"
done

test -f "${DATA_VERSION_DIR}/dataset_info.json"
test "${current_files}" -ge "${EXPECTED_DATA_FILES}"

echo "[stage 2/4][$(date -Iseconds)] Verify training stack"
conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import json
import os
from pathlib import Path

import flash_attn
import tensorflow as tf
import torch

data_root_dir = Path(os.environ["DATA_ROOT_DIR"])
dataset_info_path = data_root_dir / "custom_finetuning" / "1.0.0" / "dataset_info.json"
info = json.loads(dataset_info_path.read_text())
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu_count", torch.cuda.device_count())
print("tensorflow", tf.__version__)
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
print("dataset_info", dataset_info_path)
print("dataset_keys", sorted(info.keys()))
PY

echo "[stage 3/4][$(date -Iseconds)] Prefetch 7B checkpoint and base LLM"
export LOCAL_MODEL_ROOT BASE_LLM_REPO CRONUS_REPO_ID CRONUS_CKPT_NAME
conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

local_root = Path(os.environ["LOCAL_MODEL_ROOT"])
base_repo = os.environ["BASE_LLM_REPO"]
cronus_repo = os.environ["CRONUS_REPO_ID"]
ckpt_name = os.environ["CRONUS_CKPT_NAME"]

cronus_dir = local_root / "cronusvla_7b_bridge_rt_1"
base_dir = local_root / "llama2_7b_base"

snapshot_download(
    repo_id=cronus_repo,
    local_dir=str(cronus_dir),
    local_dir_use_symlinks=False,
    allow_patterns=["config.json", "config.yaml", "dataset_statistics.json", f"checkpoints/{ckpt_name}"],
    resume_download=True,
    max_workers=4,
)
snapshot_download(
    repo_id=base_repo,
    local_dir=str(base_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8,
)
checkpoint_path = cronus_dir / "checkpoints" / ckpt_name
if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
required_base_files = [base_dir / "config.json", base_dir / "tokenizer.json", base_dir / "tokenizer.model"]
missing = [str(path) for path in required_base_files if not path.exists()]
if missing:
    raise FileNotFoundError(f"missing base repo files: {missing}")
print("checkpoint_path", checkpoint_path)
print("base_dir", base_dir)
PY

export PRETRAINED_CHECKPOINT="${LOCAL_MODEL_ROOT}/cronusvla_7b_bridge_rt_1/checkpoints/${CRONUS_CKPT_NAME}"
export CRONUSVLA_LLAMA2_7B_HF_HUB_PATH="${LOCAL_MODEL_ROOT}/llama2_7b_base"

echo "[stage 4/4][$(date -Iseconds)] Launch training"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
conda run --no-capture-output -n "${CONDA_ENV}" bash "${RUN_SCRIPT}" 2>&1 | tee -a "${RUN_ROOT_DIR}/${JOB_NAME}.log"
