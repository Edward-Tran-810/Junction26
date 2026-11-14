#!/bin/bash
# Shared helpers for the three Isaac Sim demos. Source it; do not run it.
#   Needs a RUNNING Aalto sim and the instance env of that sim (set AALTO_INSTANCE etc. before calling, see README.md).
#   One policy server (this tree) + one evaluator; a single evaluator per sim instance at a time.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/miniforge3/etc/profile.d/conda.sh && conda activate isaac
cd "$ROOT"
export PYTHONPATH="$ROOT"
STAMP=$(date -u +%Y%m%d_%H%M%S)

# demo_run <name> <task> <seed> <policy_mode classical|rl> <extra evaluate args...>
# env: SERVER_ENV="K=V K2=V2" extra env for the policy server; PORT (default 8014 / 8013 for rl)
demo_run() {
  local name=$1 task=$2 seed=$3 mode=$4; shift 4
  local port=${PORT:-8014}; [ "$mode" = rl ] && port=${PORT:-8013}
  OUT="$ROOT/runs/demos/${name}_${STAMP}"; mkdir -p "$OUT"
  (
    export PORT=$port CUDA_VISIBLE_DEVICES=; unset AALTO_RL_GRASP AALTO_RL_SWAP_RIGHT_IM AALTO_RL_LIMITS
    if [ "$mode" = rl ]; then export AALTO_RL_SWAP_RIGHT_IM=1 AALTO_RL_GRASP="${RL_ACTOR:-/root/long/aaltonomous-humanoid-rl-grasp/runs/rl/r01_base_s1/export_model_399/policy.pt}"; fi
    for kv in ${SERVER_ENV:-}; do export "$kv"; done
    exec python -m aaltonomous.baselines.classical.policy_server > "$OUT/policy_server.log" 2>&1
  ) &
  SERVER_PID=$!
  for i in $(seq 1 60); do ss -ltn | grep -q ":$port " && break; sleep 1; done
  echo "[demo] $name: server pid $SERVER_PID port $port; sim instance '${AALTO_INSTANCE:-default}'"
  python -m aaltonomous.sim.evaluate --policy ws://localhost:$port --team "$name" --task "$task" --trials 1 --seed "$seed" \
      --out "$OUT" --video "$@" 2>&1 | tee "$OUT/evaluate.log" | grep -E "^trial|objects correct|SIM GATE|eval.json|Traceback|Error"
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  RUN_DIR=$(ls -d "$OUT"/${name}_* 2>/dev/null | head -1)
}

# demo_finish <name> <title> <subtitle> [parsed-mapping-line] : verdict + annotated mp4 + screenshots
demo_finish() {
  local name=$1 title=$2 sub=$3 extra=${4:-}
  python -m aaltonomous.demos.report "$RUN_DIR" "$OUT/policy_server.log" --title "$title" --subtitle "$sub" --extra "$extra" --out "$OUT"
}
