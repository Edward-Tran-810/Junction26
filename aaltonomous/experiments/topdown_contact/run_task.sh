#!/usr/bin/env bash
# Full-task live run (inference only) on the streaming 'gc' sim: the controller exactly as configured, scene.yaml layout.
# usage: run_task.sh <name> <task: sort_fixed|sort_language> <seed> [extra evaluate args...]
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$REPO"
NAME=$1; TASK=$2; SEED=$3; shift 3
R=runs/topdown_contact/live/$NAME
[ -e "$R" ] && { echo "$R exists: runs are never overwritten"; exit 1; }
mkdir -p "$R"
source ${SIM_ENV:-runs/topdown_contact/sim_gc_td.env}
source ~/miniforge3/etc/profile.d/conda.sh; conda activate isaac
export PYTHONPATH="$REPO" TD_PORT=${TD_PORT:-8112}
PORT=$TD_PORT CUDA_VISIBLE_DEVICES="" AALTO_TD_DIAG=1 AALTO_TD_LOG="$REPO/$R/td_picks.jsonl" \
  setsid python -m aaltonomous.baselines.classical.policy_server > "$R/policy_server.log" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null || true' EXIT
for i in $(seq 90); do ss -ltn | grep -q ":$TD_PORT " && break; sleep 1; done
python -m aaltonomous.sim.evaluate --policy ws://localhost:$TD_PORT --team topdown --task "$TASK" --trials 1 --seed "$SEED" \
  --out "$R" "$@" > "$R/evaluate.log" 2>&1 || echo "evaluate exit $?"
grep -E "^trial" "$R/evaluate.log" || true
python -m aaltonomous.experiments.topdown_contact.analyse_test "$R" > "$R/analysis.txt" 2>&1 || true
