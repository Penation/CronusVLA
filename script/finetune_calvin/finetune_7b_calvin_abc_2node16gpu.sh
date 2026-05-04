#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-/mnt/cronusvla_blob/2026vla/.hf-cache-cronusvla}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/mnt/cronusvla_blob/2026vla/.torch-cache-cronusvla}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CRONUSVLA_HF_TOKEN="${CRONUSVLA_HF_TOKEN:-}"

DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/cronusvla_blob/2026vla/calvin_abc_tfds}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/cronusvla_blob/2026vla/runs/calvin_abc_7b_2node16gpu}"
RUN_ID="${RUN_ID:-${JOB_NAME:-cronusvla-calvin-abc-7b-2node16gpu-5epoch}}"
NODES="${NODES:-${AMLT_OUTER_WORLD_SIZE:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${AMLT_OUTER_RANK:-${RANK:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
INNER_MASTER_PORT="${INNER_MASTER_PORT:-29518}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-$((NODES * GPUS_PER_NODE))}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$RUN_ROOT_DIR"

cd "$ROOT_DIR"

echo "launch_config nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} node_rank=${NODE_RANK} master_addr=${MASTER_ADDR} inner_master_port=${INNER_MASTER_PORT} expected_world_size=${EXPECTED_WORLD_SIZE}"

torchrun \
  --nnodes "${NODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${INNER_MASTER_PORT}" \
  --nproc-per-node "${GPUS_PER_NODE}" \
  training/train.py \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT:-JeasLee/cronusvla_7B_bridge_rt_1}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size "${EXPECTED_WORLD_SIZE}" \
  --vla.global_batch_size "${GLOBAL_BATCH_SIZE:-64}" \
  --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-2}" \
  --vla.learning_rate "${LEARNING_RATE:-2e-5}" \
  --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE:-4096}" \
  --vla.epochs "${EPOCHS:-5}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --image_aug True \
  --save_interval "${SAVE_INTERVAL:-20000}" \
  --repeated_diffusion_steps "${REPEATED_DIFFUSION_STEPS:-4}" \
  --future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE:-15}" \
  --past_action_window_size "${PAST_ACTION_WINDOW_SIZE:-6}" \
  --action_model_type "${ACTION_MODEL_TYPE:-DiT-B}" \
  --action_dim "${ACTION_DIM:-7}" \
  --hf_token CRONUSVLA_HF_TOKEN \
  --wandb_project "${WANDB_PROJECT:-CronusVLA}" \
  --wandb_entity "${WANDB_ENTITY:-892948933}" \
  --is_resume False
