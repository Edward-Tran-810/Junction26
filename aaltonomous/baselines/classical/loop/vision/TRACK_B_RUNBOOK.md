# Track B runbook: Isaac frames (B1), baseline vs oracle (B2), end to end (B3)

Status 2026-11-14: **not run**. Track B needs a human-approved sim slot (PERCEPTION_MILESTONES.md rule 8).

## Why manh cannot just run it

- Isaac's Python env is `/root/miniforge3/envs/isaac`, and user `manh` cannot read `/root`. The sim client side
  (`unitree_sdk2py` DDS, `teleimager`, `zmq`) also only exists in that env.
- Two sims are already running as root, from long's checkouts (checked with `ss -ltn` / `ps`, 2026-11-14):
  - instance **a**: `/root/long/aaltonomous-humanoid`, ports 55555-7 / 55560 / 55580-1, DDS domain 1
  - instance **b**: `/root/long/aaltonomous-humanoid-grasp-variants`, ports 56555-7 / 56560 / 56580-1, DDS domain 7
- **Do not connect B1/B2 clients to either.** `capture_sim.py` calls `randomize(seed)` + `env.reset()` and
  `evaluate.py` resets every trial: that would corrupt long's runs. Also the DR setting is fixed at sim launch.

## Option 1 (recommended): a third, isolated instance on GPU 0, run by someone with root

Ask long (or the admin) for a ~2 h slot. Whoever owns the slot runs, from a checkout of **manh's branch**
(`/workspace/worktrees/manh/lang-vision-loop`, so the v2 perception and `capture_sim.py` are the ones measured):

```bash
cd /workspace/worktrees/manh/lang-vision-loop
export CUDA_VISIBLE_DEVICES=0                      # GPU 1 = long's RL training
export AALTO_INSTANCE=c AALTO_DDS_DOMAIN=9 AALTO_CAM_PORT_BASE=57555 AALTO_SCENE_PORT=57580 \
       AALTO_RGBD_PORT=57560 AALTO_TELEIMAGER_REQ_PORT=61200
# third_party/unitree_sim_isaaclab must exist in this worktree (or symlink it from /root/long/... read-only)
AALTO_DR=train TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh      # in tmux; wait for "scene :57580"
```

Then every client below runs in a second shell **with the same exports** (so it talks to instance c) and
`/root/miniforge3/envs/isaac/bin/python` as `$PY`.

Note: `run_sim.sh` writes `/tmp/teleimager_aalto_c.yaml` from the a/b template and only rewrites ports 55555-7 and
6000x; check the generated file and the sim log for clashes with a and b before starting clients.

### B1. Frames with ground truth (~30 min per family at ~0.65x real time)

```bash
M=aaltonomous.baselines.classical.loop.vision
$PY -m $M.capture_sim --check-only                                     # ping, DR status, intrinsics
$PY -m $M.capture_sim --out aaltonomous/baselines/classical/loop/vision/recorded/isaac_r0 --n 420   # DR train
```
- First check that the sim renders with `interface/calibration.yaml` intrinsics (fx 604.8, cx 325.0): STATE.md
  flags that an instance started before 2026-11-12 may not. A fresh instance c does.
- Nominal family (DR off) and held-out-style family need a restart of instance c with `AALTO_DR` unset
  (`--allow-no-dr`) and with `AALTO_DR=eval_heldout` (or a yaml pushed beyond `sim/dr/train.yaml`) and a different
  `--out` each. (`eval_heldout` is our stand-in, NOT the organisers' secret variant.)
- Then, offline and without the sim, by manh:
  `python -m $M.dataset --split isaac_dev` ... `eval_vision --split isaac_fresh_01 --variant legacy v2 --overlays 5`.

### B2. What vision costs (same slot; 4 cells, 10 trials each, ~2-3 min per trial)

```bash
# oracle (true poses, same planner)
$PY -m aaltonomous.sim.evaluate --policy oracle --task sort_language --trials 10 --seed 1000 --out runs/b2/oracle_dr
# baseline with v2 perception (manh's branch) as a policy server
$PY -m aaltonomous.baselines.classical.policy_server &            # port 8000
$PY -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --task sort_language --trials 10 --seed 1000 --out runs/b2/v2_dr
```
Repeat both with the sim restarted with DR off. Same seeds (1000-1009) in all four cells.
baseline - oracle = vision (+ language) cost; oracle misses = grasp cost. Table goes into CONTEXT.md.

### B3. End to end

Same as the B2 baseline command with the chosen perception + parser, on the same seeds, under DR train and the
held-out-style DR. Then `docker build -f aaltonomous/docker/baseline.Dockerfile ...` and
`python -m aaltonomous.deploy.replay --synthetic 100 --container <image> --gpus none` for the 27 Hz gate, startup
time and image size (A0 limits).

## Option 2: share instance a or b in an agreed window

Only if long pauses their runs on it. Restart is still needed to switch DR (train / off / held-out-style).

## Option 3: give manh a runtime

Make the isaac env readable/executable for group `workspace-dev` (or a separate copy for manh). This is a
coordinated runtime change (CONTEXT.md "Runtime boundary"): admin decision, not done here.
