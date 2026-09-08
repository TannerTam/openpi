#!/bin/bash
set -euo pipefail

# Fine-tune pi0.5 on the OXE-AugE augmented Bridge dataset, using ONLY the WidowX source view
# (observation.images.image + observation.state). Single-node run on this machine's 8x A800-80GB.
#
# Adapted from run_train_table30v2_ur5_arrange_fruits_stage1_16gpu_260626.sh (that was a 2-node /
# 16-GPU multi-node job); here everything is single node, so the JAX-distributed plumbing is dropped.

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
# LeRobot resolves repo_id ("bridge_train_0_5000_augmented") under HF_LEROBOT_HOME.
export HF_LEROBOT_HOME=/home/test/test12/tanner/embodied_data/oxe-auge
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
# Dataset: 170417 frames -> ceil(170417/512) = 333 steps/epoch.
# Default 20 epochs -> 6660 steps; checkpoint every 5 epochs. Adjust EPOCHS/PER_CARD_BATCH to taste.
NUM_GPUS=8
PER_CARD_BATCH=64
EPOCHS=20
GLOBAL_BATCH=$(( PER_CARD_BATCH * NUM_GPUS ))          # 512
TOTAL_FRAMES=170417
STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))   # 333
NUM_TRAIN_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))        # ~9990
WARMUP_STEPS=1000
SAVE_INTERVAL=$(( STEPS_PER_EPOCH * 5 ))               # checkpoint every 5 epochs
NUM_WORKERS=8

exp_name="pi05_bridge_aug_widowx_${NUM_GPUS}gpu_${EPOCHS}ep_260708"
log_file="/home/test/test12/tanner/openpi/_logs/train_${exp_name}.log"

# TensorBoard logs to a persistent folder (viewable with `tensorboard --logdir <dir>`).
export OPENPI_TENSORBOARD_DIR="/home/test/test12/tanner/openpi/_tensorboard/${exp_name}"
mkdir -p "${OPENPI_TENSORBOARD_DIR}"
echo "TensorBoard logdir (persistent): ${OPENPI_TENSORBOARD_DIR}"

echo "pi05_bridge_aug_widowx finetune: ${NUM_GPUS} gpus (FSDP), global_bs=${GLOBAL_BATCH} (${PER_CARD_BATCH}/card), steps=${NUM_TRAIN_STEPS} (${EPOCHS} epochs, ${STEPS_PER_EPOCH}/epoch), save every ${SAVE_INTERVAL}, exp_name=${exp_name}"

.venv/bin/python scripts/train.py pi05_bridge_aug_widowx \
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
  --overwrite 2>&1 | tee "${log_file}"
