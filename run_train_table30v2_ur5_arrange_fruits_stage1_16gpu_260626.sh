#!/bin/bash
set -euo pipefail

# Stage 1 of the two-stage experiment: pretrain pi0.5 base on the Table30v2 (UR5)
# arrange_fruits dataset. Stage 2 (Franka arrange_fruits) initializes from the
# checkpoint produced by this run.

cd /user/tanxiyuan/openpi

# Raise the open-file-descriptor limit. Multi-node NCCL opens a large number of
# sockets across the two nodes; the default soft limit (often 1024) caused the
# previous run to crash on the first collective with
#   "socketTryAccept: Accept failed: Too many open files".
# Running as root, so first try to raise the hard limit, then raise the soft
# limit up to it (fallback to a large fixed value, then to the current hard limit).
ulimit -Hn 1048576 2>/dev/null || true
hard_nofile="$(ulimit -Hn 2>/dev/null || echo 1048576)"
[ "${hard_nofile}" = "unlimited" ] && hard_nofile=1048576
ulimit -n 1048576 2>/dev/null || ulimit -n "${hard_nofile}" 2>/dev/null || true
echo "open file limit (nofile): $(ulimit -n)"

# Two-node launch: run one Python process per machine, each controlling its
# local 8 GPUs. JAX distributed will combine the two processes into 16 devices.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [ -n "${WORLD_SIZE:-}" ] && [ "${WORLD_SIZE}" -gt 1 ]; then
  : "${RANK:?RANK must be set for multi-node JAX training}"
  : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node JAX training}"
  export JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES:-${WORLD_SIZE}}"
  export JAX_PROCESS_ID="${JAX_PROCESS_ID:-${RANK}}"
  export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-${MASTER_ADDR}:${MASTER_PORT:-12355}}"
fi
# JAX warned that this proxy env var "may cause a hang of distributed.initialize";
# unset it defensively for multi-node NCCL stability.
unset NCCL_PROXY_DUMP_SIGNAL 2>/dev/null || true
export HF_LEROBOT_HOME=/user/tanxiyuan/cache/huggingface/lerobot
export OPENPI_DATA_HOME=/user/tanxiyuan/cache/openpi_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export JAX_COMPILATION_CACHE_DIR=/user/tanxiyuan/cache/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p /user/tanxiyuan/openpi/_logs

# NOTE: keep this exp_name in sync with the stage-2 weight_loader path in config.py
# (pi05_franka_arrange_fruits_ft_table30) and in the stage-2 launch script.
exp_name="table30v2_ur5_arrange_fruits_16gpu_10ep_260626"
process_id="${JAX_PROCESS_ID:-${RANK:-0}}"
log_file="/user/tanxiyuan/openpi/_logs/train_table30v2_${exp_name}_process${process_id}.log"

# TensorBoard logs are written to YOUR persistent folder so they survive job teardown
# and are viewable anytime via `tensorboard --logdir <dir>` in cursor/vscode.
export OPENPI_TENSORBOARD_DIR="/user/tanxiyuan/openpi/_tensorboard/${exp_name}"
mkdir -p "${OPENPI_TENSORBOARD_DIR}"
# Best-effort: also expose the same logs to the cybertron task page "Open TensorBoard"
# button by symlinking the running job's logs/tensorboard/<exp> to the persistent dir.
job_id=$(hostname | grep -oE '[0-9]+' | head -1 || true)
job_tb_root=$(ls -d /projects/*/"${job_id}"/logs/tensorboard 2>/dev/null | head -1 || true)
[ -n "${job_tb_root}" ] && ln -sfn "${OPENPI_TENSORBOARD_DIR}" "${job_tb_root}/${exp_name}" || true
echo "TensorBoard logdir (persistent): ${OPENPI_TENSORBOARD_DIR}"

# 2 machines x 8 GPUs, global batch 768 (48/card). ~3.19M frames -> ~4154 steps/epoch.
# 3 epochs -> 12462 steps; save every epoch (4154 steps) -> 3 checkpoints.
echo "Stage 1: Table30v2 UR5 arrange_fruits pretrain: 2x8 gpus, process=${process_id}/${JAX_NUM_PROCESSES:-${WORLD_SIZE:-1}}, coordinator=${JAX_COORDINATOR_ADDRESS:-none}, global_bs=768 (48/card), steps=12462 (3 epochs), save every 4154 (1 epoch), exp_name=${exp_name}"
.venv/bin/python scripts/train.py pi05_table30v2_ur5_arrange_fruits \
  --exp-name "${exp_name}" \
  --checkpoint-base-dir /user/tanxiyuan/openpi/checkpoints \
  --batch-size 768 \
  --fsdp-devices 16 \
  --num-train-steps 12462 \
  --lr-schedule.warmup-steps 1000 \
  --lr-schedule.decay-steps 12462 \
  --save-interval 4154 \
  --keep-period 4154 \
  --num-workers 8 \
  --overwrite 2>&1 | tee "${log_file}"
