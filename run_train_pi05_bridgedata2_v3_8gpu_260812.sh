#!/bin/bash
set -euo pipefail

# Fine-tune pi0.5 on the local BridgeData2 LeRobot v3 dataset (delta-EEF action space).
# Single-node run on this machine's 8x A800-80GB.
#
# Adapted from run_train_pi05_bridge_aug_widowx_8gpu_260708.sh.

cd /home/test/test12/tanner/openpi

# Raise the open-file-descriptor limit (data workers + many mmapped parquet/video handles).
ulimit -Hn 1048576 2>/dev/null || true
hard_nofile="$(ulimit -Hn 2>/dev/null || echo 1048576)"
[ "${hard_nofile}" = "unlimited" ] && hard_nofile=1048576
ulimit -n 1048576 2>/dev/null || ulimit -n "${hard_nofile}" 2>/dev/null || true
echo "open file limit (nofile): $(ulimit -n)"

# Single node, all 8 local GPUs.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- Environment for this machine ---------------------------------------------------------------
# LeRobot resolves repo_id ("BridgeData2_LeRobot_v3") under HF_LEROBOT_HOME.
export HF_LEROBOT_HOME=/home/test/test12/tanner/embodied_data
# openpi asset/checkpoint cache; the pi05_base weights the weight_loader needs live here under
# openpi-assets/checkpoints/pi05_base/params, so training can run fully offline.
export OPENPI_DATA_HOME=/home/test/test12/tanner/checkpoints
# This dataset stores camera frames as MP4 (LeRobot v3.0). torchcodec needs system FFmpeg libs that
# aren't installed here, so decode via the self-contained pyav backend (bundled with the av wheel).
export OPENPI_VIDEO_BACKEND=pyav
# github/network is flaky on this box; stay offline (base weights + dataset are already local).
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export JAX_COMPILATION_CACHE_DIR=/home/test/test12/tanner/openpi/_jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p /home/test/test12/tanner/openpi/_logs "${JAX_COMPILATION_CACHE_DIR}"

# --- Run / batching knobs -----------------------------------------------------------------------
# 8 GPUs, fully sharded (FSDP). per-card batch 64 -> global batch 512.
# 30k steps total, checkpointed (and kept) every 5k steps, per the requested schedule.
NUM_GPUS=8
PER_CARD_BATCH=64
GLOBAL_BATCH=$(( PER_CARD_BATCH * NUM_GPUS ))          # 512
NUM_TRAIN_STEPS=30000
WARMUP_STEPS=1000
SAVE_INTERVAL=5000
NUM_WORKERS=8

exp_name="pi05_bridgedata2_v3_${NUM_GPUS}gpu_30k_260812"
log_file="/home/test/test12/tanner/openpi/_logs/train_${exp_name}.log"

# TensorBoard logs to a persistent folder (viewable with `tensorboard --logdir <dir>`).
export OPENPI_TENSORBOARD_DIR="/home/test/test12/tanner/openpi/_tensorboard/${exp_name}"
mkdir -p "${OPENPI_TENSORBOARD_DIR}"
echo "TensorBoard logdir (persistent): ${OPENPI_TENSORBOARD_DIR}"

echo "pi05_bridgedata2_v3 finetune: ${NUM_GPUS} gpus (FSDP), global_bs=${GLOBAL_BATCH} (${PER_CARD_BATCH}/card), steps=${NUM_TRAIN_STEPS}, save every ${SAVE_INTERVAL}, exp_name=${exp_name}"

.venv/bin/python scripts/train.py pi05_bridgedata2_v3 \
  --exp-name "${exp_name}" \
  --checkpoint-base-dir /home/test/test12/tanner/openpi/checkpoints \
  --batch-size "${GLOBAL_BATCH}" \
  --fsdp-devices "${NUM_GPUS}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --lr-schedule.warmup-steps "${WARMUP_STEPS}" \
  --lr-schedule.decay-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-period "${SAVE_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --resume 2>&1 | tee -a "${log_file}"
