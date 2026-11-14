#!/bin/bash
# Run a list of "<run.yaml> <seed>" lines (stdin or file) with K concurrent trainings on GPU 1.
#   bash aaltonomous/rl/launch.sh K jobs.txt [max_iterations]
K=${1:-2}; JOBS=$2; IT=${3:-}
cd "$(dirname "$0")/../.."
source ~/miniforge3/etc/profile.d/conda.sh && conda activate isaac
export PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
mkdir -p runs/rl
while read -r run seed; do
  [ -z "$run" ] && continue
  while [ "$(pgrep -fc 'aaltonomous.rl.train ')" -ge "$K" ]; do sleep 30; done
  id=$(basename "$run" .yaml)
  echo "$(date +%T) start $id s$seed"
  nohup python -m aaltonomous.rl.train --run "$run" --seed "$seed" ${IT:+--max-iterations $IT} \
     > "runs/rl/${id}_s${seed}.log" 2>&1 &
  sleep 60
done < "$JOBS"
wait
