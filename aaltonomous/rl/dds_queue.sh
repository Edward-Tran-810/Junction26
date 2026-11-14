#!/bin/bash
# Queued DDS evaluation: wait until fewer than 2 Aalto sims run (the classical agent may use up to 2), start an
# isolated instance (AALTO_INSTANCE=c, DDS domain 9, ports 575xx), run one command against it, stop the sim.
#   bash aaltonomous/rl/dds_queue.sh <log_dir> python -m aaltonomous.sim.grasp_test --out ... [args]
# The command gets the instance env vars (AALTO_DDS_DOMAIN, AALTO_*_PORT) so AaltoEnv / SceneClient find this sim.
set -u
LOG="$1"; shift
mkdir -p "$LOG"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AALTO_INSTANCE=c AALTO_DDS_DOMAIN=9 AALTO_CAM_PORT_BASE=57555 AALTO_SCENE_PORT=57580 AALTO_RGBD_PORT=57560
export AALTO_TELEIMAGER_REQ_PORT=62000 TASK=Aalto-Sort-G129-Dex3-Joint
n_sims() { pgrep -f "^python .*aalto_sim_main" | wc -l; }
echo "[queue] $(date +%T) waiting for < 2 sims (now $(n_sims))" | tee -a "$LOG/queue.log"
while [ "$(n_sims)" -ge 2 ]; do sleep 60; done
echo "[queue] $(date +%T) starting sim instance c" | tee -a "$LOG/queue.log"
CUDA_VISIBLE_DEVICES=1 setsid bash "$REPO/aaltonomous/sim/run_sim.sh" > "$LOG/sim.log" 2>&1 &
sleep 5
SIMPID=$(pgrep -f "^python .*aalto_sim_main" -n)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate isaac
export PYTHONPATH="$REPO"
for i in $(seq 1 60); do   # up to 10 min
  python -c "
from aaltonomous.sim.scene_client import SceneClient
import os; c = SceneClient('127.0.0.1'); print(c.ping())" > /dev/null 2>&1 && break
  sleep 10
done
echo "[queue] $(date +%T) sim up (pid $SIMPID), running: $*" | tee -a "$LOG/queue.log"
cd "$REPO" && "$@" > "$LOG/run.log" 2>&1
echo "[queue] $(date +%T) command exit $?" | tee -a "$LOG/queue.log"
kill -INT "$SIMPID" 2>/dev/null; sleep 20; kill -9 "$SIMPID" 2>/dev/null
echo "[queue] $(date +%T) sim stopped" | tee -a "$LOG/queue.log"
