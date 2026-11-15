"""Item 4: thumb / table clearance and the safe stopping distance of the guarded descent, on REAL planner output
(grasp.mode=topdown) with the full Dex3 collision meshes (not the planner's sparse hand points) and the real palm tilt.
For each object position: plan a pick (mock detection), take the grasp pose (end of the guarded descent) and every tick of
the descent / close phases, place the surface-sampled Dex3 link meshes there (palm FK of the G1 model x hand FK of the
Dex3 URDF) and measure the lowest hand point above the counter. Stopping distance = peak hand speed in the guarded
descent x detection latency (policy chunk + confirm messages + one command tick + PD lag allowance).
Run: python -m aaltonomous.experiments.topdown_contact.clearance_check"""
import json, numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.experiments.topdown_contact.posture_check import Hand
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

MIRROR = np.array([1.0, -1.0, 1.0])

def left7(side, h14):
    """Hand posture of `side` in LEFT joint convention (the Dex3 URDF used by Hand is the left hand)."""
    if side == "left":
        return np.asarray(h14[:7], float)
    r = np.asarray(h14[7:], float)
    return -np.array([r[0], r[1], r[2], r[5], r[6], r[3], r[4]])

def main():
    cfg = load_config(); sc = Scene.load(); spec = JointSpec(); post = hand_postures(cfg, spec); home = home_arm(cfg, spec)
    pl = Planner(cfg, sc, post, home, spec=spec); kin = pl.kin; hand = Hand(600); zc = sc.counter_z
    td = cfg["grasp"]["topdown"]; g = td["guard"]
    keys = [k for k in hand.pts if k != "palm"]
    out = []
    for shape in ("cube", "cylinder", "triangle"):
        H = sc.object_height(shape)
        for x, y in ((0.28, 0.09), (0.28, -0.09), (0.34, 0.0), (0.34, 0.15), (0.40, -0.12)):
            d = Detection(color="red", top=np.array([x, y, zc + H]), n_px=500, n_depth=400, method="depth",
                          yaw=None if shape == "cylinder" else 0.0, shape=shape, shape_conf=0.5)
            pl.drop_count = {}
            p = pl.plan(d, home.copy(), post["open"].copy(), relocate=False)
            if not p.ok:
                out.append(dict(shape=shape, xy=[x, y], ok=False, reason=p.reason)); continue
            ph = {n: (s, e) for n, s, e in p.phases}
            side = p.side
            lows, tilts = [], []
            for name in ("descend", "close"):
                s, e = ph[name]
                for k in range(s, e, 2):
                    kin._fk(p.arm[k]); M = kin.data.oMf[kin.palm[side]]
                    hq = left7(side, p.hand[k])
                    P = np.vstack(list(hand.posed(hq, keys).values()))        # left-hand palm frame
                    if side == "right":
                        P = P * MIRROR                                     # right hand = mirror image in palm y
                    W = P @ M.rotation.T + M.translation
                    lows.append((name, k, float(W[:, 2].min() - zc)))
                    tilts.append(float(np.degrees(np.arccos(np.clip(-(M.rotation @ np.array([0, -1.0 if side == 'left' else 1.0, 0]))[2], -1, 1)))))
            # guarded descent speed (grasp point, m/s)
            s, e = ph["descend"]
            z = np.array([kin.grasp_pose(side, p.arm[k])[0][2] for k in range(s, e)])
            v = float(np.abs(np.diff(z)).max() * 30.0) if len(z) > 1 else 0.0
            latency = int(g.get("chunk", 2)) / 30.0 + int(g.get("confirm_msgs", 3)) / 100.0 + 1 / 30.0 + 0.05
            dmin = min(l[2] for l in lows if l[0] == "descend"); cmin = min(l[2] for l in lows if l[0] == "close")
            out.append(dict(shape=shape, xy=[x, y], ok=True, side=side, descend_s=round((e - s) / 30.0, 2),
                            descend_speed_mm_s=round(v * 1e3, 1), latency_s=round(latency, 3),
                            stop_distance_mm=round(v * latency * 1e3, 1), lowest_hand_mm_descend=round(dmin * 1e3, 1),
                            lowest_hand_mm_close=round(cmin * 1e3, 1), max_palm_tilt_deg=round(max(tilts), 1),
                            palm_gap_mm=(p.grasp or {}).get("palm_gap_mm")))
            print(out[-1], flush=True)
    open("aaltonomous/experiments/topdown_contact/clearance_report.json", "w").write(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
