#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-/Data/rlds_raw/.hf-cache-cronusvla}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/Data/rlds_raw/.torch-cache-cronusvla}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

hf_token="${HF_TOKEN:-$(cat /home/v-wangxiaofa/.cache/huggingface/token)}"
export CRONUSVLA_HF_TOKEN="$hf_token"
export WANDB_MODE="${WANDB_MODE:-offline}"

pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-JeasLee/cronusvla_0.5B_bridge_rt_1}"
data_root_dir="${DATA_ROOT_DIR:-/Data/rlds_raw/task_ABC_D/calvin_abc_tfds}"
run_root_dir="${RUN_ROOT_DIR:-/Data/rlds_raw/cronusvla_runs/calvin_abc_qwen0_5b}"
run_id="${RUN_ID:-cronusvla-calvin-abc-qwen0_5b-4gpu}"

nproc_per_node="${NPROC_PER_NODE:-4}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-1}"
global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
max_steps="${MAX_STEPS:-50000}"
save_interval="${SAVE_INTERVAL:-1000}"
learning_rate="${LEARNING_RATE:-2e-5}"
shuffle_buffer_size="${SHUFFLE_BUFFER_SIZE:-4096}"
future_action_window_size="${FUTURE_ACTION_WINDOW_SIZE:-15}"
past_action_window_size="${PAST_ACTION_WINDOW_SIZE:-3}"
repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS:-4}"
action_model_type="${ACTION_MODEL_TYPE:-DiT-B}"
action_dim="${ACTION_DIM:-7}"

mkdir -p "$run_root_dir"

cd "$ROOT_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node "$nproc_per_node" training/train.py \
  --pretrained_checkpoint "$pretrained_checkpoint" \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size "$nproc_per_node" \
  --vla.global_batch_size "$global_batch_size" \
  --vla.per_device_batch_size "$per_device_batch_size" \
  --vla.learning_rate "$learning_rate" \
  --vla.max_steps "$max_steps" \
  --vla.shuffle_buffer_size "$shuffle_buffer_size" \
  --data_root_dir "$data_root_dir" \
  --run_root_dir "$run_root_dir" \
  --run_id "$run_id" \
  --image_aug True \
  --save_interval "$save_interval" \
  --repeated_diffusion_steps "$repeated_diffusion_steps" \
  --future_action_window_size "$future_action_window_size" \
  --past_action_window_size "$past_action_window_size" \
  --action_model_type "$action_model_type" \
  --action_dim "$action_dim" \
  --hf_token CRONUSVLA_HF_TOKEN \
  --is_resume False
