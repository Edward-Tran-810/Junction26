# topdown-contact — STATE

Integrated into the `integration/grasp-fixes` worktree (uncommitted, local only; nothing pushed / merged). The offline
analysis was first done in worktree `topdown-contact/r0`. grasp.mode is now `topdown` (config.yaml).

Note: `integration/grasp-fixes` records a project decision (2026-11-14) "SIDE GRASP ONLY, legacy top-down disabled".
The 2026-11-15 request asks for TOP-DOWN ONLY with contact-triggered closure. Work happens on this separate branch;
the side-grasp config on integration is not modified.

## Step 1 — Dex3 geometry + contact sensing (offline, DONE 2026-11-15)

Source: official `unitree_dex3_left.urdf` collision geometry (STL meshes; thumb_1 = box) + pinocchio FK,
surface-sampled links, analytic 45 mm cube / cylinder / triangular prism (scene.yaml `objects.size`).

- Palm frame (left): x = finger direction, fingers and thumb curl towards **-y**, so the contact surface is the
  **-y pad at y = -0.0202 m** (palm link origin = wrist side, it is NOT the contact surface). Pad ~ x 0.003-0.068,
  z +-0.03 (within 2 mm of the lowest point).
- The **thumb root link (thumb_0) protrudes 8 mm below the pad** (y = -0.0283) at x 0.014-0.036.
  `contact_link_check.py`: the palm pad is the first link to touch the object top only for object centre x >= 0.060.
- **Physical palm-pad-on-top + 3-finger closure is INFEASIBLE** for the 45 mm objects (`closure_search.py`, 3000
  random preshape/target pairs per finger group in URDF limits, table plane included): with the palm pad on the
  top the thumb (~98 mm long, root 23 mm from the wrist, curls only towards the object) never reaches the object
  before hitting the table; for every shape (cube, cylinder, triangle 0/180 deg, fallen cylinder). Without the table
  (`geometry_check.py`) it "works", which is why the table must be modelled.
- **Near-contact** (separately labelled mode, NOT physical palm contact): with the palm pad 20 mm above the top
  and object centre at palm x = 0.070 (cube), thumb contacts the back face 4 mm above the table and index /
  middle the front face 11-13 mm above the table = opposed wrap. 10 mm gap: only with the object in front of the palm.
  Reports: `closure_gap_cube.json`, `closure_gap_shapes.json`.
  Per shape (1500 samples/group; "opposed" = thumb on the back face, fingers on the front face):
  gap 0.5 mm (= palm contact): no shape feasible. 15 mm: cylinder, triangle@180 (apex towards the wrist), and the
  fallen cylinder (both axes) at palm x 0.080 only. 20 mm: cube and cylinder at 0.070-0.080, triangle@180 at 0.080,
  fallen cylinder at 0.070-0.080 (axis along x) / 0.080 (axis along z). triangle@0 (apex towards the fingers):
  never opposed up to 25 mm. The prism has to be yawed so that its apex points to the wrist.
- Contact sensing: no tactile data in the sim's Dex3 DDS state. Robot USD has `activate_contact_sensors=True`.
  Added (opt-in, `AALTO_HAND_CONTACTS=1`, default scene unchanged): one Isaac Lab `ContactSensor` per Dex3 link
  (palm, thumb_1/2, index_0/1, middle_0/1, both hands), each filtered against every object; published per env step
  (100 Hz) as JSON on zmq PUB `AALTO_CONTACT_PORT` (default scene port + 2): per link net normal force,
  force per object, `other` = net - objects (table / lid / bins). Not yet run in a sim.
- Actuators: sim Dex3 = implicit PD kp 100, kd 10, **effort_limit 300** (upstream preset), while the URDF limits are
  **2.45 N m (thumb_0) / 1.4 N m (others)**. The sim does not enforce the real limits, so the controller must:
  closing target - measured q <= tau_max / kp per joint.

## Open decisions (for the human)
1. Palm-contact closure is geometrically infeasible -> proceed with the labelled near-contact mode (+ contact
   monitoring for unexpected collisions and per-finger closure contacts)?
2. Machine load ~60 on 60 cores with other agents' sims/evals running; a new sim instance (planned: instance
   `tc`, DDS domain 11, ports 585xx) will slow their evals (CLAUDE.md: eval gate fails below 27 Hz).

## Step 2 - top-down controller (2026-11-15, integration worktree)

- planner.py: `grasp.mode: topdown` = near-contact top-down (palm level, max_tilt 8 deg; fast approach, then a slow
  guarded descent of guard_height; 25 mm verification lift `lift`, rest `lift2`; one `_lift_path`). `side` and
  `topdown_legacy` (env-gated) kept. Same IK / counter / bin-rim / safety dry-run checks for every mode.
- topdown_exec.py (policy hooks): PRE-GRASP -> DESCENDING (guarded) -> trigger -> CLOSING (lead-limited: command =
  measured + clip(goal - measured, L), L = 0.85 tau_max / kp = 0.0119 rad = 1.19 N m) -> CLOSED (hold frozen) -> small
  lift -> GRASP VERIFIED (finger load) -> LIFTING -> carry -> RELEASE (ramp from the actual command) -> release check.
  Triggers recorded separately: `palm_contact` (verified, PhysX) | `fallback_link_contact` | `fallback_height`.
- `guard.contacts_for_control: false` (default): PhysX contacts are diagnostic only; the descent guard is proprioceptive
  (hand joint error growth vs the start of the descent > 0.03 rad). Nothing privileged drives a decision.
- Diagnostics (AALTO_TD_DIAG=1, sim only, never control): object pose rel. to the grasp point (slip / rotation / rise),
  model thumb clearance from measured joints, per-link contact events; per-pick JSONL via AALTO_TD_LOG.
- Sims: `tc` (ours, contacts, no stream) used for t1-t4, stopped after t6. `gc` (ours) restarted idle with contacts +
  WebRTC livestream (sim_gc_td.env), t5+ run there (visible in the user's viewer).

## Physical cube tests (Isaac, seed 3000 unless noted; runs/topdown_contact/<test>/summary.json + mp4)

| test | change | result |
|---|---|---|
| t1 | first run (gap 22) | 2/2 attempts: thumb-table contact 1.6 / 2.4 N -> safe abort. Plan had 29 deg palm tilt |
| t2 | max_tilt 8 deg | tilt 0: thumb-table 3.1 / 2.8 N -> abort |
| t3 | height logging | thumb-table 2.8 / 3.6 N at measured pad 65.4 mm (cmd 69.8, plan 67): ~4 mm arm sag |
| t4 | gap 32 mm | no collision. #1 thumb+index 16/20 N, cube stayed on the table at the lift (empty). #2 SORTED: 3 contacts, rise 92 mm, slip 10.9 mm, rot 23 deg, release clean |
| t5 | contacts diagnostic-only, proprio guard | guard false trip: thumb_2 measured 1.658 vs cmd 1.72 (sim joint stop) = constant offset |
| t6 | guard baseline + td_pre thumb_2 1.65 (reachable) | SORTED first attempt, no table contact, 3 contacts at verify, rise 85 mm, slip 9.7 mm, rot 10.3 deg, clean release. Gate FAIL rt 24.4 Hz (two sims + diag queries) |

| t7 | t6 config, seed 3001 | SORTED 1st attempt, no table contact, thumb model clearance 6.9 mm, slip 3.1 mm, rot 2.1 deg, gate ok |
| t8 | t6 config, seed 3002 | SORTED 1st attempt, no table contact, 6.8 mm, slip 4.7 mm, rot 4.1 deg, gate ok |
| t9 | cylinder (t6 config) | FAILED 2/2 (safe aborts, no table contact): thumb never reached the cylinder; every closing so far ended on the 3 s timeout |
| t10 | prism (t6 config) | SORTED on attempt 2 (#1 no contact at all), 3 contacts 20/13/12 N, slip 8.8 mm, rot 23 deg |
| t11 | squeeze.max_s 3 -> 6 s, cylinder | SORTED 1st attempt, 3 contacts 25/13/17 N, slip 3.9 mm, rot 10.3 deg, gate ok |
| t12 | max_s 6, prism | SORTED 1st attempt, thumb+index 17/19 N (3 at verify), slip 3.4 mm, rot 10.1 deg, gate ok |
| t13 | max_s 6, cube (regression check) | SORTED 1st attempt, 3 contacts 21/21/18 N, slip 2.8 mm, rot 11.2 deg, 0 regrips, gate ok |

Thumb collision root cause: colliding link = left_hand_thumb_2_link (lowest link, 63.9 mm below the palm pad at the
real tuck). Sim thumb_2 stops 0.062 rad short of 1.72 (+2.8 mm lower) + ~4.4 mm arm sag vs a 7.3 mm URDF margin at gap
22 mm -> ~0. Fix: reachable thumb_2 1.65 command + gap 32 mm (model margin 15.9 - 4.4 sag ~ 11 mm, no contact in t4-t6).
thumb_1 further back LOWERS the thumb (no help). Open issue: contacts 32-37 mm up the 45 mm faces -> slip / rotation.
Fallen cylinder: the scene only places objects upright (yaw); not testable without changing the official scene.
Task 3 frozen (user, 2026-11-15).
