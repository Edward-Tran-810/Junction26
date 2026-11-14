#!/bin/bash
# Start the lid scene (Aalto-Insert task) as sim instance 'd' from THIS worktree and wait until its scene service answers.
# Resources: one more Isaac sim (CPU physics, ~7 GB RAM, ~4 GB VRAM on GPU $GPU, default 0). Stop it with:  bash aaltonomous/demos/stop_sim_insert.sh
#   bash aaltonomous/demos/start_sim_insert.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/aaltonomous/demos/instance_d.env"
GPU=${GPU:-0}
pgrep -af "aalto_sim_main" | grep -q "AALTO_INSTANCE=d" && true
for p in $(pgrep -f "^python .*aalto_sim_main"); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=d" && { echo "instance d already running (pid $p)"; exit 0; }
done
mkdir -p "$ROOT/runs/demos"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$GPU TASK=Aalto-Insert-G129-Dex3-Joint setsid nohup bash aaltonomous/sim/run_sim.sh > runs/demos/sim_d.log 2>&1 < /dev/null &
source /root/miniforge3/etc/profile.d/conda.sh && conda activate isaac && export PYTHONPATH="$ROOT"
t0=$SECONDS
for i in $(seq 1 240); do
  python -m aaltonomous.sim.scene_client ping >/dev/null 2>&1 && { echo "instance d ready after $((SECONDS - t0)) s"; exit 0; }
  sleep 5
done
echo "instance d did NOT come up; tail of runs/demos/sim_d.log:"; tail -20 runs/demos/sim_d.log; exit 1
