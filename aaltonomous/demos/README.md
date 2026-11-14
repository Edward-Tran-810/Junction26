# Isaac Sim demos (local branch integration/grasp-fixes)

Run from the repo root (`/root/long/aaltonomous-humanoid-integration`). Each script starts its own policy server, runs ONE episode
through `sim/evaluate.py` (real robot model, controllers and physics), writes `runs/demos/<name>_<ts>/`:
`<name>_<ts>.mp4` (head camera + text overlay), `screenshots/*.png` (frames of that video), `result.json` (PASS / FAIL / BLOCKED from the
simulator's final scene), `policy_server.log`, `evaluate.log`, and the evaluator's `eval.json` + per-tick `episode_000.npz`.
The MP4 is the robot's HEAD CAMERA. The Kit viewport is not captured; to watch live use the WebRTC livestream of sim b (owned by the
grasp-variants agent; see aaltonomous/baselines/classical/variants/STATE.md) and run the demos on b.

Instance variables for sim b (the Sort sim with the livestream):
    export AALTO_INSTANCE=b AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560

    bash aaltonomous/demos/demo1_fixed_sort.sh 2000 classical      # red->left yellow->center blue->right
    bash aaltonomous/demos/demo2_instruction.sh                    # unseen instruction, mapping blue=left red=center yellow=right
    bash aaltonomous/demos/demo2_instruction.sh "Blue goes right, red in the middle, yellow left." "blue=right,red=center,yellow=left" 2002
    bash aaltonomous/demos/demo1_fixed_sort.sh 2000 rl             # same with the RL residual (existing checkpoint)

Demo 3 (lid) needs the Aalto-Insert scene = its own sim instance d (uses ~7 GB RAM, ~4 GB VRAM, CPU physics):
    bash aaltonomous/demos/start_sim_insert.sh                     # starts d, waits ~5 min
    bash aaltonomous/demos/demo3_insert.sh 2000 classical
    bash aaltonomous/demos/stop_sim_insert.sh

Lid mesh: aaltonomous/scene/assets/lid_shapes.stl (210 x 210 x 3 mm, 55 mm square / 55 mm circle / 60 mm triangle holes). The mesh named
lid_shape_holes_200mm.stl was NOT available on this machine: copy it to aaltonomous/scene/assets/lid_shape_holes_200mm.stl and tell us; the sim,
scene.lid_holes() and the scorer must then be switched to it together (scene.yaml lid.* and sim/aalto_task/scene_cfg.lid_cfg).

Pose of the lid: official orientation (AALTO_LID_ROT_DEG=0). AALTO_LID_ROT_DEG=180 turns it on its box (diagnostic only, NOT a valid demo
because it changes the challenge geometry).
