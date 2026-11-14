# RL grasp track: state

Round: **0 done except the dense DDS check (queued); round 1 training started 2026-11-14 12:04.**
GPU-hours used (GPU 1 wall-clock with >= 1 of my jobs): ~1.0 before training (env build, replays, benchmark);
training log in `gpu_hours.log`. Budget 40.
Branch `rl-grasp`, worktree `../aaltonomous-humanoid-rl-grasp` (`third_party` = symlink to the main checkout).
RL library: **rsl_rl 5.5.1** (PPO, asymmetric actor/critic).

## What exists (aaltonomous/rl/)
| file | what |
|---|---|
| `grasp_env.py` | `Aalto-Grasp-G129-Dex3-RL` DirectRLEnv: scene pieces imported from `sim/aalto_task/scene_cfg.py`; physics of `env_cfg.py`; 30 Hz targets held 8/6/6 physics steps (= DDS timing); one active object per env (3 per env, 2 parked); residual on the script, projected onto 97 % of the velocity caps + filter joint range (0 clips by construction); DR: friction / mass / finger model sampled once per env |
| `scripted.py` | oracle planner (ORACLE_OVERRIDES + face-aligned grasp, imported) for the LEFT arm -> pregrasp..lift + 1 s hold + 1 s shake (2 Hz, shoulder pitch/roll + wrist yaw, <= 0.75 rad/s) + 0.5 s; library `data/lib_left.npz` = 3 shapes x 5 x x 8 y x 13 yaws, 1560/1560 planned |
| `cells.py` | library for explicit cells (the grasp_test grid; right-arm cells mirrored onto the left arm) |
| `evaluate.py` | replay scripted / policy on library or cells, physics + noise presets, `--device cpu`, `--bench` |
| `train.py`, `launch.sh` | rsl_rl PPO; K concurrent runs on GPU 1; logs + checkpoints in repo-root `runs/rl/` (gitignored) |
| `skill.py` | deployment: CPU TorchScript actor, right side mirrored (mirror map checked by FK: 0.01 mm), projection to caps; not wired in yet |
| `dds_queue.sh` | waits until < 2 Aalto sims run, starts instance `c` (DDS domain 9, ports 575xx), runs one command, stops it |
| `_paths.py` | imports upstream `tasks.common_config` WITHOUT `tasks/__init__` (that starts a DDS participant on domain 1 = the live eval sim) |

## Round 0 measurements
**Throughput** (GPU 1, random actions, one process): 256 envs 2 921 env-steps/s, 1024: 10 570, 4096: 26 992 (6.6 ticks/s).

**Scripted grasp replayed in the training env** (success = lifted >= 10 cm for the whole 1 s hold AND after the shake,
no keep-out violation; counts = successes / episodes):
| set | physics | cube | cylinder | triangle |
|---|---|---|---|---|
| library 1560, no noise | fingers = object 1.2 | 508/520 | 513/520 | 171/520 |
| library subset 312, CPU PhysX | same | 100/104 | 104/104 | 31/104 (GPU same cases 26/104; 33/104 triangle cases flip GPU<->CPU) |
| grasp_test cells, 20 jittered repeats (+-1 mm, +-1 deg) | fingers = 1.2 | 390/400 | 343/400 | 166/400 |
| grasp_test cells, same | **dds_a0** = eval-sim materials (fingers mu 5) | 387/400 | 303/400 | 113/400 |
| grasp_test cells, start from home instead of pregrasp | fingers = 1.2 | 381/400 | 349/400 | 130/400 |
| **DDS grasp_test, stock oracle, A0** (classical agent's rows, 20 cells x 3 repeats) | eval sim | 58/60 | 48/60 | 28/60 |

**Validation verdict.** Cube and cylinder agree within 6 points on the exact cells in both finger materials.
Triangle: 42 % (fingers 1.2) / 28 % (eval-sim materials) vs 47 % DDS. The DDS reference has only 20 distinct
triangle cells and is near-deterministic per cell (3/3 or 0/3), so its 95 % interval is about +-22 points; by arm
the split is reversed (DDS left 18/30, right 10/30; env left 41/200, right 72/200 in dds_a0). Triangle outcomes are
chaotic per cell (33 % flip between CPU and GPU PhysX in the same env). **Not resolved**: queued a denser DDS run
(triangle + cylinder x 7 yaws x 4 lateral x 2 depth = 112 distinct cells) to settle it. **Decision (mine, for review):**
training started before that result because cube/cylinder agree, the bugs found were fixed, and every checkpoint
must pass DDS grasp_test anyway. The first large disagreement (33 % vs 47 % triangle, 99 % vs 80 % cylinder) was my
error: different grid (library region vs grasp_test cells) and my "nominal" set the fingers to 1.2 instead of the
eval sim's mu 5.

Ruled out as causes (measured): GPU vs CPU PhysX (aggregates within 5 points), the start state (pregrasp vs home),
the bin the plan carries to (changes the grasp in 4/60 cells), the arm assignment (oracle's own selection = inferred
for 60/60 cells), colour cycle (60/60 match).

## Findings that matter beyond this track
1. **The DDS sim swaps the right hand's index and middle.** Upstream `action_provider_dds.py` maps
   `rt/dex3/right/cmd` as thumb 0-2, middle 3-4, index 5-6 (the LEFT order); INTERFACE.md (measured on the robot):
   right = thumb, index, middle. Harmless for the classical postures (index = middle values), wrong for anything that
   moves them differently (e.g. an RL residual on the right hand). Not patched by `aalto_sim_main.py`.
2. Eval-sim finger friction is mu 5.0 except the left index (0.5) (found by the classical agent, reproduced here).
   With max-combine, the scripted grasp is BETTER with all fingers at 1.2 than with the asset's mu 5 (training env,
   grasp_test cells: triangle 166/400 vs 113/400, cylinder 343 vs 303). Same direction as the classical A6 vs A0.
3. Sim Dex3 thumb_1 lower limit is -0.611 (USD) vs -0.724 (URDF / joint_spec).

## Policy design as built (deviations from the brief, with reasons)
- Action: residual on left arm 7 + left hand 7 (`full`), or hand 7 (`hand`), or hand + 3 wrist joints
  (`hand_wrist`); |residual| <= 0.12 rad (arm) / 0.5 rad (hand), then projected: change per tick <= 0.97 x cap,
  inside the filter's joint range. **Keep-out/workspace check uses measured wrist/palm link positions with a 1 cm
  margin** (a proxy for the filter's FK of the commanded target; DDS evaluation checks the real thing).
- Actor obs (68 + action dim): left arm q, dq, hand q (+ noise), the script's next target (14), phase one-hot +
  progress, last action, object estimate (shape one-hot, xy, yaw harmonics 1/3/4) = the pose the script was planned
  for. The TRUE object = estimate + uniform noise (spec: +-5 mm, +-10 deg, 5 % wrong shape) -> the script itself
  follows the noisy estimate, as with real perception.
- Critic: + object pose/velocity relative to the grasp point, rise, fingertip offsets, true shape, mu, mass, gains.
- Reward: +10 success (end of episode), +0.05/tick x clamp(rise / 10 cm) after descend, -0.02 |a_t - a_t-1|^2,
  -0.2/tick fingertip within 3 mm of the counter, -1 keep-out violation.
- Friction "narrow" = object 1.2 x [0.75, 1.1] with ALL robot shapes set to the same value (contact 0.9-1.3);
  the eval sim's mu-5 fingers are a held-out condition (`dds_a0`) instead of the training default. Reason: finding 2.

## Run matrix (one factor at a time around r01; 2 seeds each)
| id | bets on | scripted baseline in its own training distribution (cube / cyl / tri, n ~1040 each) |
|---|---|---|
| r01_base | nominal +-, perception as spec, full residual | 1009/1039, 904/1028, 291/1053 |
| r02_fric_wide | friction anywhere 0.3-1.2 | 1007/1039, 703/1028, 101/1053 |
| r03_fingers_rand | Dex3 gains x0.5-2 / damping x0.5-2 / armature x0.2-2 / joint friction 0-0.05 | 948/1039, 864/1028, 292/1053 |
| r04_robust | r02 + r03 | 947/1039, 638/1028, 118/1053 |
| r05_act_hand | hand-only residual | = r01 |
| r06_act_handwrist | hand + wrist residual | = r01 |
| r07_noise_low | +-2 mm, +-4 deg, no wrong shape | 998/1040, 980/1040, 255/1040 |
| r08_noise_high | +-10 mm, +-20 deg, 10 % wrong shape, joint noise 0.01 | 974/1053, 707/1019, 290/1048 |
| r09_no_estimate | proprioception + script only | = r01 |
(r02/r04 baselines were measured before the finger-material fix; wide mode was unaffected.)

Training runs: pilot 1 (std 0.3, white noise) collapsed to success 0 (measured: scripted + N(0, s) actions,
r01 distribution: s=0.05 cyl 68/511; s=0.1 cube 28/531, cyl 0/511). Fixes, each measured:
- residual = EMA of actions (alpha 0.2): s=0.1 -> cube 457/531, cyl 308/511, tri 61/518. Noise on the arm only
  is harmless (509 / 430 / 128); the hand residual is the sensitive part.
- init std 0.1, gamma 0.995, counter penalty only below 1 mm (w 0.05).
- Pilot 2 (r01 s1, 60 iterations, 7.9 M steps): deterministic eval in r01's own distribution, 3120 cases, same cases
  and physics draws as the scripted baseline: cube 980/1039 (scripted 1009), cyl 862/1028 (904), tri 312/1053 (291).

**Round 1 started 13:33**: 9 configs x 2 seeds x 400 iterations (52 M steps each), 2 concurrent on GPU 1
(~14k fps each). `autoeval.py` evaluates model_200 and model_399 deterministically in the run's own distribution
vs the scripted baseline on the same cases (kill rule), and model_399 also on held-out physics (fric0.3, fric0.9,
mass1.5, the other finger model, dds_a0). r01 s2 died at start (a path under /workspace that vanished, not code);
re-queued at the end.

## Integration (step 3 prep, measured offline)
- Hook: commit `1d44593`, 11 lines in `classical/policy.py`, off unless `cfg["rl_grasp"]` or `$AALTO_RL_GRASP`;
  test_classical.py 10/10 OK with it off.
- `export.py`: TorchScript actor + `skill_meta.json`; CPU 1 thread: 0.020 ms median, 0.035 ms p99 per step.
- `check_skill.py` (pilot-2 actor, kinematic robot, real SafetyFilter incl. FK workspace/keep-out): 60/60 grasp_test
  cells planned, **0 clipped steps**, max |residual| 0.277 rad. The skill imports no Isaac/pinocchio modules.
- `AALTO_RL_SWAP_RIGHT_IM=1` (DDS sim only) compensates the sim's right-hand index/middle swap.

## Questions for the human
1. Triangle validation (above): I started training before the dense DDS result. Stop me if you want it first.
2. The classical agent's A4 (real Dex3 kp 1.5 / kd 0.2, ~67x softer than the sim) was dropped by you for that
   track. My "random finger model" (x0.5-2) does not reach it. Should an RL run bet on the soft hand?
3. Right-hand index/middle swap in the DDS sim (finding 1): fix upstream mapping in the shim, or leave it?
4. GPU-hours: I count GPU 1 wall-clock while any of my jobs runs (concurrent runs share it). Tell me if you meant
   per-process hours.
