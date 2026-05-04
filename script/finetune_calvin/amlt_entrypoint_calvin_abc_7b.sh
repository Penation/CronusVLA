#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_DIR}/script/finetune_calvin/finetune_7b_calvin_abc_2node16gpu.sh}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:?DATA_ROOT_DIR is required}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:?RUN_ROOT_DIR is required}"
CONDA_ENV="${CONDA_ENV:-cronusvla}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/scratch/cronusvla_models}"
BASE_LLM_REPO="${BASE_LLM_REPO:-NousResearch/Llama-2-7b-hf}"
CRONUS_REPO_ID="${CRONUS_REPO_ID:-JeasLee/cronusvla_7B_bridge_rt_1}"
CRONUS_CKPT_NAME="${CRONUS_CKPT_NAME:-step-055000-epoch-04-loss=0.0286.pt}"
NODES="${NODES:-${WORLD_SIZE:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29517}"
INNER_MASTER_PORT="${INNER_MASTER_PORT:-29518}"
OUTER_RANK="${RANK:-0}"
OUTER_WORLD_SIZE="${WORLD_SIZE:-1}"

DATA_VERSION_DIR="${DATA_ROOT_DIR}/custom_finetuning/1.0.0"
EXPECTED_DATA_FILES="${EXPECTED_DATA_FILES:-1158}"
DATA_WAIT_SECONDS="${DATA_WAIT_SECONDS:-60}"
DATA_WAIT_MAX_POLLS="${DATA_WAIT_MAX_POLLS:-360}"

echo "[stage 1/5][$(date -Iseconds)] Validate code package and wait for blob dataset"
echo "outer_rank=${OUTER_RANK} outer_world_size=${OUTER_WORLD_SIZE} nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} master_addr=${MASTER_ADDR:-unset} master_port=${MASTER_PORT} inner_master_port=${INNER_MASTER_PORT}"
pwd
nvidia-smi
df -h /scratch /mnt/cronusvla_blob 2>/dev/null || true
test -f "${RUN_SCRIPT}"
test -f "${REPO_DIR}/training/train.py"
mkdir -p "${RUN_ROOT_DIR}"

current_files=0
for ((poll_idx=1; poll_idx<=DATA_WAIT_MAX_POLLS; poll_idx++)); do
  current_files="$(find "${DATA_VERSION_DIR}" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -f "${DATA_VERSION_DIR}/dataset_info.json" && "${current_files}" -ge "${EXPECTED_DATA_FILES}" ]]; then
    echo "dataset_root=${DATA_ROOT_DIR}"
    echo "dataset_files=${current_files}/${EXPECTED_DATA_FILES}"
    find "${DATA_VERSION_DIR}" -maxdepth 1 -type f | sort | sed -n '1,20p'
    break
  fi

  echo "waiting_for_dataset poll=${poll_idx} files=${current_files}/${EXPECTED_DATA_FILES}"
  sleep "${DATA_WAIT_SECONDS}"
done

test -f "${DATA_VERSION_DIR}/dataset_info.json"
test "${current_files}" -ge "${EXPECTED_DATA_FILES}"

echo "[stage 2/5][$(date -Iseconds)] Build CronusVLA conda env"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
cd "${REPO_DIR}"
bash script/setup_train_env.sh "${CONDA_ENV}"

echo "[stage 3/5][$(date -Iseconds)] Verify remote training stack"
conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import json
from pathlib import Path
import os

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

splits = info.get("splits", {})
if isinstance(splits, dict):
    split_items = splits.items()
elif isinstance(splits, list):
    split_items = ((split.get("name", f"split_{idx}"), split) for idx, split in enumerate(splits))
else:
    split_items = []

for split_name, split_info in split_items:
    num_examples = split_info.get("numExamples")
    num_bytes = split_info.get("numBytes")
    shard_lengths = split_info.get("shardLengths", [])
    print(
        "split",
        split_name,
        "num_examples",
        num_examples if num_examples is not None else "unknown",
        "num_bytes",
        num_bytes if num_bytes is not None else "unknown",
        "num_shards",
        len(shard_lengths),
    )
PY

echo "[stage 4/5][$(date -Iseconds)] Prefetch 7B checkpoint and public base LLM to local scratch"
mkdir -p "${LOCAL_MODEL_ROOT}" "${HF_HOME}" "${TORCH_HOME}"
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

print("prefetch_cronus_repo", cronus_repo, "->", cronus_dir)
snapshot_download(
    repo_id=cronus_repo,
    local_dir=str(cronus_dir),
    local_dir_use_symlinks=False,
    allow_patterns=["config.json", "config.yaml", "dataset_statistics.json", f"checkpoints/{ckpt_name}"],
    resume_download=True,
    max_workers=4,
)

print("prefetch_base_repo", base_repo, "->", base_dir)
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

required_base_files = [
    base_dir / "config.json",
    base_dir / "tokenizer.json",
    base_dir / "tokenizer.model",
]
missing = [str(path) for path in required_base_files if not path.exists()]
if missing:
    raise FileNotFoundError(f"missing base repo files: {missing}")

for label, path in [("checkpoint_path", checkpoint_path), ("base_dir", base_dir)]:
    size_gb = 0.0
    if path.is_file():
        size_gb = path.stat().st_size / (1024 ** 3)
    else:
        size_gb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 ** 3)
    print(label, path, f"{size_gb:.2f} GiB")
PY

export PRETRAINED_CHECKPOINT="${LOCAL_MODEL_ROOT}/cronusvla_7b_bridge_rt_1/checkpoints/${CRONUS_CKPT_NAME}"
export CRONUSVLA_LLAMA2_7B_HF_HUB_PATH="${LOCAL_MODEL_ROOT}/llama2_7b_base"
export NODES GPUS_PER_NODE MASTER_PORT INNER_MASTER_PORT
export AMLT_OUTER_RANK="${OUTER_RANK}"
export AMLT_OUTER_WORLD_SIZE="${OUTER_WORLD_SIZE}"

echo "local_pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
echo "local_base_llm_dir=${CRONUSVLA_LLAMA2_7B_HF_HUB_PATH}"

echo "[stage 5/5][$(date -Iseconds)] Launch CronusVLA CALVIN ABC finetune"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
if [[ "${OUTER_RANK}" == "0" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" bash "${RUN_SCRIPT}" 2>&1 | tee -a "${RUN_ROOT_DIR}/${JOB_NAME}.log"
else
  conda run --no-capture-output -n "${CONDA_ENV}" bash "${RUN_SCRIPT}" 2>&1 | tee -a "${RUN_ROOT_DIR}/${JOB_NAME}.node${OUTER_RANK}.log"
fi
