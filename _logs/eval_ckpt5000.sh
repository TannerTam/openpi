#!/bin/bash
# Re-run the SimplerEnv eval for checkpoint 5000, which failed earlier when the server's
# first-request JIT compilation outlived the websocket keepalive timeout.
set -uo pipefail

cd /home/test/test12/tanner/openpi

ckpt=5000
PORT=8000

OPENPI_DATA_HOME=/home/test/test12/tanner/checkpoints \
HF_LEROBOT_HOME=/home/test/test12/tanner/embodied_data \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
nohup .venv/bin/python scripts/serve_policy.py --port ${PORT} policy:checkpoint \
  --policy.config pi05_bridgedata2_v3 \
  --policy.dir /home/test/test12/tanner/openpi/checkpoints/pi05_bridgedata2_v3/pi05_bridgedata2_v3_8gpu_30k_260812/${ckpt} \
  > _logs/serve_simpler_${ckpt}.log 2>&1 &
server_pid=$!
echo "server pid: ${server_pid}"

for i in $(seq 1 60); do
  if grep -q "server listening" _logs/serve_simpler_${ckpt}.log 2>/dev/null; then
    echo "server is up (waited ${i}x2s)"
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "server process died while loading, see _logs/serve_simpler_${ckpt}.log"
    exit 1
  fi
  sleep 2
done
sleep 3

echo "=== checkpoint ${ckpt}: running eval client ==="
/home/test/test12/miniconda3/envs/tanner_iglt_simpler/bin/python examples/simpler_env/main.py \
  --port ${PORT} --task all --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --max-steps 120 --video-dir eval_outputs/simpler_env_${ckpt} \
  > _logs/eval_simpler_${ckpt}.log 2>&1
echo "checkpoint ${ckpt} eval exit code: $?"
tail -n 8 _logs/eval_simpler_${ckpt}.log

kill -9 "${server_pid}" 2>/dev/null
echo "CKPT ${ckpt} EVAL COMPLETE"
