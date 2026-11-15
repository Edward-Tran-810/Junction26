"""Offline planner check of grasp.mode=topdown (near-contact top-down): plans over a grid of object positions /
yaws / shapes (sort_fixed bins), counts feasible plans and failure reasons. No sim.
Run: python -m aaltonomous.experiments.topdown_contact.plan_grid [--insert]"""
import argparse, collections, json, numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--insert", action="store_true"); ap.add_argument("--overrides", default="{}")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    cfg = load_config(overrides=json.loads(a.overrides)); sc = Scene.load(); spec = JointSpec()
    post = hand_postures(cfg, spec); home = home_arm(cfg, spec)
    pl = Planner(cfg, sc, post, home, spec=spec)
    hand0 = post["open"].copy(); zc = sc.counter_z
    rows, cnt = [], collections.Counter()
    for shape in ("cube", "cylinder", "triangle"):
        H = sc.object_height(shape)
        for x in (0.28, 0.34, 0.40, 0.46):
            for y in (-0.18, -0.09, 0.0, 0.09, 0.18):
                for yaw in ((0.0,) if shape == "cylinder" else (0.0, np.radians(25), np.radians(-30))):
                    for color in ("red", "yellow", "blue"):
                        if a.insert and color != "red": continue
                        d = Detection(color=color, top=np.array([x, y, zc + H]), n_px=500, n_depth=400, method="depth",
                                      yaw=None if shape == "cylinder" else yaw, shape=shape, shape_conf=0.5)
                        pl.drop_count = {}
                        p = pl.plan_insert(d, home.copy(), hand0.copy()) if a.insert else pl.plan(d, home.copy(), hand0.copy(), relocate=False)
                        r = (p.reason or "") if not p.ok else "OK"
                        import re; key = re.sub(r"[-+]?\d+(\.\d+)?", "#", r)[:70]
                        cnt[(shape, key)] += 1
                        rows.append(dict(shape=shape, xy=[x, y], yaw_deg=round(float(np.degrees(yaw)), 0), color=color, ok=bool(p.ok),
                                         reason=p.reason, side=p.side, grasp=p.grasp))
    for shape in ("cube", "cylinder", "triangle"):
        n = sum(v for (s, k), v in cnt.items() if s == shape); ok = cnt[(shape, "OK")]
        print(f"{shape:9s} plans ok {ok}/{n}")
        for (s, k), v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            if s == shape and k != "OK": print(f"     {v:3d}  {k}")
    if a.out: open(a.out, "w").write(json.dumps(rows, indent=1, default=str))

if __name__ == "__main__":
    main()
