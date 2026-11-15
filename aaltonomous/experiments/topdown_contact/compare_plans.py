"""Offline: planned phase durations + feasibility, baseline vs a variant (AALTO_CFG_OVERRIDES-style JSON), same positions.
Run: python -m aaltonomous.experiments.topdown_contact.compare_plans runs/topdown_contact/variants/fast_approach_v1.json"""
import json, sys, collections, numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

def run(over):
    cfg = load_config(overrides=over); sc = Scene.load(); spec = JointSpec(); post = hand_postures(cfg, spec); home = home_arm(cfg, spec)
    pl = Planner(cfg, sc, post, home, spec=spec); zc = sc.counter_z
    dur, ok, n, reasons = collections.defaultdict(list), 0, 0, collections.Counter()
    for shape in ("cube", "cylinder", "triangle"):
        for x in (0.27, 0.31, 0.35):
            for y in (-0.16, -0.08, 0.0, 0.08, 0.16):
                d = Detection(color="red" if y > 0.05 else ("blue" if y < -0.05 else "yellow"), top=np.array([x, y, zc + sc.object_height(shape)]),
                              n_px=500, n_depth=400, method="depth", yaw=None if shape == "cylinder" else 0.3, shape=shape, shape_conf=.5)
                pl.drop_count = {}; n += 1
                p = pl.plan(d, home.copy(), post["open"].copy(), relocate=False)
                if not p.ok:
                    reasons[(p.reason or "")[:60]] += 1; continue
                ok += 1
                for name, s, e in p.phases:
                    dur[name].append((e - s) / 30.0)
                dur["TOTAL"].append(len(p.arm) / 30.0)
    return ok, n, {k: round(float(np.mean(v)), 2) for k, v in dur.items()}, reasons

if __name__ == "__main__":
    over = json.load(open(sys.argv[1]))
    for name, o in (("baseline", {}), ("variant", over)):
        ok, n, d, r = run(o)
        print(f"{name:9s} plans ok {ok}/{n}  mean phase s: {d}")
        for k, v in r.most_common(4): print(f"          fail {v}x {k}")
