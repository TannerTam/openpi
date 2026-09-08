#!/bin/bash
# Evaluates the Bridge suite driven by the Google Robot, placed with OXE-AugE's offset.
# Stops the policy server when done so the GPU is released.
set -uo pipefail

cd /home/test/test12/tanner/openpi

SERVER_PID=1897061

/home/test/test12/miniconda3/envs/tanner_iglt_simpler/bin/python examples/simpler_env/main.py \
  --suite bridge --robot google_robot --task all \
  --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --diagnose --video-dir eval_outputs/bridge_google_auge_25000 \
  > _logs/eval_bridge_google_auge.log 2>&1
echo "eval exit code: $?"

grep -aE "^widowx.*[0-9]+/[0-9]+ = |^overall:|observed m|train  q" _logs/eval_bridge_google_auge.log

kill -9 "${SERVER_PID}" 2>/dev/null && echo "stopped policy server ${SERVER_PID}"
sleep 2
echo "remaining serve_policy processes: $(pgrep -c -f serve_policy.py || echo 0)"
echo "EVAL COMPLETE"
