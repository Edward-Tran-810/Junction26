"""Offline check of the deployment path (no sim): plan every grasp_test cell with the oracle planner for the arm the
oracle uses, run the plan through ClassicalPolicy's queue with the RL skill hooked in (measured = commanded, i.e. a
kinematic robot), filter every tick through a fresh SafetyFilter (joint limits, velocity caps, workspace + counter
keep-out via FK). Reports clips, the max residual, inference time.

    python -m aaltonomous.rl.check_skill --policy runs/rl/<run>/export_model_399/policy.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--swap-right", action="store_true")
    a = ap.parse_args(argv)
    import os
    if a.swap_right:
        os.environ["AALTO_RL_SWAP_RIGHT_IM"] = "1"
    from aaltonomous.baselines.classical.perception import Detection
    from aaltonomous.interface.safety import SafetyFilter
    from aaltonomous.rl import scripted as S
    from aaltonomous.rl.cells import grasp_test_cells
    from aaltonomous.rl.skill import GraspSkill

    pl, base = S.make_planner()
    sc = base.scene
    skill = GraspSkill(a.policy, base.spec)
    base.skill = skill
    tot_clips, rows = 0, []
    for c in grasp_test_cells():
        H = sc.object_height(c["shape"])
        d = Detection(c["color"], np.array([c["px"], c["py"], sc.counter_z + H]), 1000, 1000, "oracle",
                      yaw=float(S.mat_yaw_to_pelvis(np.radians(c["yaw_mat"]))))
        d.shape, d.name = c["shape"], "o"
        b = {"red": "left", "yellow": "center", "blue": "right"}[c["color"]]
        p = pl.plan(d, base.home.copy(), base.post["open"].copy(), bin_name=b, relocate=False)
        if not p.ok:
            rows.append({"cell": c, "planned": False})
            continue
        base.last_detections = [d]
        base.arm_cmd, base.hand_cmd = base.home.copy(), base.post["open"].copy()
        f = SafetyFilter(base.spec, base.safety)
        f.reset(base.arm_cmd, base.hand_cmd, 0.0)
        base._enqueue(p.arm, p.hand)
        skill.begin(p, base)
        k, t, maxres = 0, 0.0, 0.0
        obs_prev = {"arm_q": base.arm_cmd.copy(), "hand_q": base.hand_cmd.copy()}
        while base.queue_a is not None and len(base.queue_a):
            dq = np.zeros(14)
            obs = {"arm_q": base.arm_cmd.copy(), "arm_dq": (base.arm_cmd - obs_prev["arm_q"]) * 30,
                   "hand_q": base.hand_cmd.copy(), "t": t}
            obs_prev = obs
            plan_next = np.r_[base.queue_a[0], base.queue_h[0]]
            act = skill.act(base, obs) if skill.active else base._pop()
            A, Hh = np.atleast_2d(act["arm_q_target"]), np.atleast_2d(act["hand_q_target"])
            if len(A) == 1:
                maxres = max(maxres, float(np.abs(np.r_[A[0], Hh[0]] - plan_next).max()))
            for i in range(len(A)):
                t += 1 / 30
                f.filter({"arm_q_target": A[i], "hand_q_target": Hh[i]}, t)
        rows.append({"cell": f"{c['shape']}/{c['yaw_mat']:.0f}/{c['x_mat']:.3f} {c['side']}", "clips": f.clipped_steps,
                     "max_residual": round(maxres, 3)})
        tot_clips += f.clipped_steps
    ms = np.array(skill.times) * 1000
    print(json.dumps({"cells": len(rows), "planned": sum(1 for r in rows if r.get("planned", True)),
                      "clipped_steps_total": tot_clips,
                      "cells_with_clips": [r for r in rows if r.get("clips")],
                      "max_residual_rad": max(r.get("max_residual", 0) for r in rows),
                      "infer_ms_median": float(np.median(ms)), "infer_ms_p99": float(np.percentile(ms, 99))},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
