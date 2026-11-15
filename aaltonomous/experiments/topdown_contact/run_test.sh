#!/usr/bin/env bash
# One physical top-down test on the isolated 'tc' sim (runs/topdown_contact/sim_tc.env, contact sensors ON).
# usage: run_test.sh <name> <shape> <seed> [task] [extra evaluate args...]
# -> runs/topdown_contact/<name>/{policy_server.log,evaluate.log,td_picks.jsonl,summary.json,<eval dir>/episode_000.mp4}
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$REPO"
NAME=$1; SHAPE=$2; SEED=$3; TASK=${4:-sort_fixed}; shift 3; [ $# -gt 0 ] && shift
R=runs/topdown_contact/$NAME
[ -e "$R" ] && { echo "$R exists: tests are never overwritten"; exit 1; }
mkdir -p "$R"
source ${SIM_ENV:-runs/topdown_contact/sim_tc.env}
source ~/miniforge3/etc/profile.d/conda.sh; conda activate isaac
export PYTHONPATH="$REPO" TD_PORT=${TD_PORT:-8111}
PORT=$TD_PORT CUDA_VISIBLE_DEVICES="" AALTO_TD_DIAG=1 AALTO_TD_LOG="$REPO/$R/td_picks.jsonl" \
  setsid python -m aaltonomous.baselines.classical.policy_server > "$R/policy_server.log" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null || true' EXIT
for i in $(seq 90); do ss -ltn | grep -q ":$TD_PORT " && break; sleep 1; done
python -m aaltonomous.sim.evaluate --policy ws://localhost:$TD_PORT --team topdown --task "$TASK" --trials 1 --seed "$SEED" \
  --n-objects 1 --shapes "$SHAPE" --time-limit 90 --video --out "$R" "$@" > "$R/evaluate.log" 2>&1 || echo "evaluate exit $?"
grep -E "^trial" "$R/evaluate.log" || true
python -m aaltonomous.experiments.topdown_contact.analyse_test "$R"
