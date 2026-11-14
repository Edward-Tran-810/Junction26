"""Vision round 0: record the frozen Isaac frame set (head RGB-D + ground-truth object poses) for offline evaluation.

!!! Do NOT run this without a human-approved Isaac time slot (another developer uses the simulator). This script
!!! never starts, restarts or stops Isaac: the sim must already be running, started by the person who owns the slot,
!!! with domain randomisation on:
!!!     AALTO_DR=train TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh
!!! and shut down by them afterwards.

    python -m aaltonomous.baselines.classical.loop.vision.capture_sim --out loop/vision/recorded/isaac_r0 --n 420
    python -m aaltonomous.baselines.classical.loop.vision.capture_sim --check-only        # ping + DR status, no frames

Per frame (one frame per layout and per DR draw, so no two frames share a scene):
  1. arms home, every object parked (sim/evaluate.home_arms)
  2. scene service `randomize(seed)` (DR parameter set; recorded) and env.reset() with env.rng = seed
     (tasks.sample_layout, as in the evaluator)
  3. wait until every active object is at rest, then a few more ticks so the head stream holds a fresh frame
  4. head_rgb + head_depth from the observation, object poses from the scene service `state` (GROUND TRUTH: used
     only offline by eval_vision.py; it is never an input to the policy or to Perception)
Output: <out>/<id>.npz (head_rgb, head_depth) + <out>/<id>.json (GT in the synthetic GT format) + <out>/index.jsonl.
Frame ids are "isaac-band-<seed>"; dataset.py serves them as "rec:<path>.npz".

Ground truth conventions (same as synth.py): centre = object root position (pelvis frame); top = centre + the
object's body z axis * height / 2; yaw = rotation about pelvis +z of the canonical outline (from the quaternion);
tilt_deg = angle between the body z axis and pelvis z (> 10 deg: knocked over, flagged). visible_px / unoccluded_px
= pixel area of the projected prism with the NOMINAL calibration (occlusion between objects is not computed: the
arms are at home, out of view); truncated = the projected prism leaves the image.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def quat_to_R(q_xyzw):
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def projected_area_px(scene, cal, shape, centre, yaw, cam="head"):
    """(pixel count of the prism silhouette, truncated?) with the nominal calibration."""
    import cv2
    from aaltonomous.baselines.classical.shape_fit import outline
    h = scene.object_height(shape)
    o = outline(scene, shape, yaw) + np.asarray(centre[:2])
    P = np.vstack([np.c_[o, np.full(len(o), centre[2] - h / 2)], np.c_[o, np.full(len(o), centre[2] + h / 2)]])
    uvz = cal.project(cam, P)
    c = cal.cams[cam]
    if np.any(uvz[:, 2] <= 0):
        return 0, True
    hull = cv2.convexHull(uvz[:, :2].astype(np.float32))[:, 0, :]
    truncated = bool(hull[:, 0].min() < 0 or hull[:, 1].min() < 0 or hull[:, 0].max() >= c.width
                     or hull[:, 1].max() >= c.height)
    img = np.zeros((c.height, c.width), np.uint8)
    cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32), 1, shift=4)
    return int(img.sum()), truncated


def gt_from_state(state_objects, scene, cal):
    """Scene-service `state()["objects"]` -> GT object list (active objects only). Pure function (unit-tested)."""
    out = []
    for o in state_objects:
        if not o.get("active", True):
            continue
        R = quat_to_R(o["quat_xyzw"])
        h = scene.object_height(o["shape"])
        centre = np.asarray(o["pos"], float)
        top = centre + R[:, 2] * h / 2
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        area, trunc = projected_area_px(scene, cal, o["shape"], centre, yaw)
        out.append({"name": o.get("name"), "color": o["color"], "shape": o["shape"],
                    "center": [round(float(v), 6) for v in centre], "top": [round(float(v), 6) for v in top],
                    "yaw": round(yaw, 6), "tilt_deg": round(tilt, 2), "tipped": tilt > 10.0,
                    "speed": float(o.get("speed", 0.0)), "visible_px": area, "unoccluded_px": area,
                    "truncated": trunc})
    return out


def capture(a):
    import ssl  # noqa: F401  before DDS (see sim/evaluate.py)
    from aaltonomous.sim.evaluate import apply_instance_env, home_arms, settle
    apply_instance_env()
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.scene.scene import Scene
    from aaltonomous.sim.scene_client import SceneClient

    sc = SceneClient(a.host)
    info = sc.info()
    print("scene service:", {k: info.get(k) for k in ("dr", "n_objects", "rtf_mean")})
    if not info.get("dr") and not a.allow_no_dr:
        raise SystemExit("the sim runs WITHOUT domain randomisation (AALTO_DR unset); round 0 needs AALTO_DR=train "
                         "(or pass --allow-no-dr for a nominal set)")
    if a.check_only:
        return
    scene, cal = Scene.load(), Calibration.load()
    rest = float(scene.cfg["success"]["at_rest_speed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=3600.0, img_server_ip=a.host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    index = out / "index.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(line)["seed"] for line in index.read_text().splitlines() if line.strip()}
    try:
        for i in range(a.n):
            seed = a.seed0 + i
            if seed in done:
                continue
            home_arms(env, sc)
            dr_params = sc.randomize(seed).get("params")
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            settle(sc, a.settle_s, rest)
            hold = {"arm_q_target": env.filter.arm_cmd.copy(), "hand_q_target": env.filter.hand_cmd.copy()}
            for _ in range(int(a.fresh_ticks)):            # let the head stream deliver frames of the settled scene
                obs, _ = env.step(hold)
            st = sc.state()
            gt_objs = gt_from_state(st["objects"], scene, cal)
            fid = f"isaac-band-{seed}"
            np.savez_compressed(out / f"{fid}.npz", head_rgb=obs["head_rgb"], head_depth=obs["head_depth"])
            gt = {"id": fid, "seed": seed, "family": "isaac", "layout_mode": "band", "layout": env.layout,
                  "objects": gt_objs, "dr_params": dr_params, "camera_error": (dr_params or {}).get("camera"),
                  "t_capture": time.time(), "valid_depth_frac": float((obs["head_depth"] > 0).mean())}
            (out / f"{fid}.json").write_text(json.dumps(gt, default=float))
            with open(index, "a") as f:
                f.write(json.dumps({"id": fid, "seed": seed, "n_objects": len(gt_objs),
                                    "tipped": sum(o["tipped"] for o in gt_objs)}) + "\n")
            print(f"[{i + 1}/{a.n}] {fid}: {len(gt_objs)} objects, depth valid {gt['valid_depth_frac']:.2f}")
    finally:
        env.close()
    print(f"frames in {out}; the Isaac slot owner shuts the sim down now")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "recorded" / "isaac_r0"))
    ap.add_argument("--n", type=int, default=420, help="frames (>= 400 required for round 0)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--settle-s", type=float, default=4.0, help="max wall seconds waiting for objects at rest")
    ap.add_argument("--fresh-ticks", type=int, default=15, help="30 Hz ticks held after settling, before the shot")
    ap.add_argument("--allow-no-dr", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="ping the scene service, report DR, record nothing")
    capture(ap.parse_args(argv))


if __name__ == "__main__":
    main()
