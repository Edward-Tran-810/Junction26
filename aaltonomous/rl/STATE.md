# RL grasp track: state

Round: **0 (in progress: reading done, env not built yet)**. GPU-hours used: **0 / 40**.
Branch `rl-grasp`, worktree `../aaltonomous-humanoid-rl-grasp` (`third_party` is a symlink to the main checkout, gitignored).

## Facts gathered in round 0 (read from code, not measured)

- Eval task physics (`sim/aalto_task/env_cfg.py`): sim.dt 0.005, decimation 2 (env step 10 ms), 16/4 min solver
  iterations, CCD on, bounce 0.01, friction_correlation_distance 0.003. Commands arrive at 30 Hz on the sim clock, so
  a target is held for 4/3/3 env steps (8/6/6 physics steps). Plan: RL env uses physics dt 0.005 with the same
  [8, 6, 6] physics-step hold pattern, which gives exactly 30 Hz with the same zero-order-hold timing as DDS.
- Robot: upstream `G129_CFG_WITH_DEX3_BASE_FIX`, self-collisions off, articulation solver 8/4.
  Arms: implicit PD, kp 300 (shoulders) / 400 (elbow, wrist), kd 3.0 / 2.5. Hands: kp 100, kd 10, armature 0.1,
  effort 300, velocity limit 100. These are the "default gains" finger model.
- Objects: convex hull of the print STLs, 50 g, friction 1.2 / 1.0 with combine mode `max` (so the object's friction
  dominates any finger contact), contact offset 4 mm. Counter: static box, friction 0.8 / 0.6.
  Because of `max` combine mode, "fingertip friction" can only be varied by lowering the object's own friction (or by
  switching the combine mode in the training env, which would make it differ from the eval env: record it if done).
- `sim/grasp_test.py` runs the **oracle** (classical planner + privileged poses), one object per episode,
  default yaws -45/0/30 (mat frame) x 4 mat-x points x middle mat-y, 22 s limit; outcome = full pick **and place**
  in the correct bin, not just lift. The RL training-env success (lift 10 cm, hold 1 s, shake) is a different metric,
  so the round-0 comparison has to report both "lifted" and "placed" from the scripted replay.
- Oracle grasp: `ORACLE_OVERRIDES` + face-aligned grasp (`FACE_ALIGN`, triangle finger angle `AALTO_TRI_PHI`=105 deg),
  pinch thumb vs one finger (`grasp.point_in_palm` = [0.099, -0.084, 0.015]), postures preshape / closed / release
  from `classical/config.yaml`. sim README: oracle ~84 % on rung 1, misses are mostly triangles (prism spins).
- Classical variants track (`../aaltonomous-humanoid-grasp-variants`): no `variants/STATE.md` or `results.jsonl`
  yet as of 2026-11-14, so there is no "best variant" to beat and no failure table. Baseline for now: the stock oracle.

## Next steps (round 0, not started)

1. `aaltonomous/rl/grasp_env.py`: DirectRLEnv `Aalto-Grasp-G129-Dex3-RL`, scene = robot + counter (+ mat collider
   not needed) + one object per env (3 shape assets, chosen per env), assets imported from `scene_cfg.py`.
2. Scripted-trajectory generator: run the oracle planner offline per (shape, pose) for the left arm, cache
   pregrasp -> descend -> close -> lift trajectories, add a scripted shake.
3. Replay the scripted grasp in the RL env across shapes / yaws / positions; compare with `grasp_test` in DDS
   (check `pgrep -f "^python .*aalto_sim_main"` first; at most 2 sims belong to the classical agent).
4. Throughput at 256 / 1024 / 4096 envs on GPU 1. RL library choice: rsl_rl (PPO, asymmetric critic supported).

## Run table

None yet.

## Questions for the human

- None blocking yet. The truncated lines in the task prompt (e.g. "holds up when physics assumptions chan...") were
  read as: robust to friction / finger-model changes; deliver ranked checkpoints, the human decides on submission.
