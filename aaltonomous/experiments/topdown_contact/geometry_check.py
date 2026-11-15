"""Offline Dex3 palm-contact feasibility (no sim): can a 45 mm object sit directly under the palm pad while the thumb,
index and middle fingers close around it?  Uses the official Dex3 collision STLs + URDF joint chain (pinocchio), surface
samples of every link, and analytic solids for cube / cylinder / triangular prism (size 0.045, scene.yaml).
Palm frame (left hand): x = finger direction, fingers + thumb curl towards -y, so the palm 'contact face' is the -y pad.
Run:  python -m aaltonomous.experiments.topdown_contact.geometry_check  [--out geometry_report.json]"""
import argparse, json, numpy as np, trimesh, pinocchio as pin
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "third_party/xr_teleoperate/assets/unitree_hand/unitree_dex3_left.urdf"
MESH = URDF.parent / "meshes"
S = 0.045
J = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
     "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint"]
LINKS = {"palm": "left_hand_palm_link", "t0": "left_hand_thumb_0_link", "t1": "left_hand_thumb_1_link", "t2": "left_hand_thumb_2_link",
         "m0": "left_hand_middle_0_link", "m1": "left_hand_middle_1_link", "i0": "left_hand_index_0_link", "i1": "left_hand_index_1_link"}

def load_pts(n=1500):
    out = {}
    for k, ln in LINKS.items():
        f = MESH / f"{ln}.STL"
        if k == "t1":   # URDF collision for thumb_1 is a box 0.02x0.03x0.02 at (-0.001,-0.032,0)
            m = trimesh.creation.box((0.02, 0.03, 0.02)); m.apply_translation((-0.001, -0.032, 0))
        else:
            m = trimesh.load(f)
        out[k] = trimesh.sample.sample_surface(m, n, seed=0)[0]
    return out

def inside(shape, p, c, yaw=0.0):      # p (N,3) in palm frame; c = solid centre; solid axis = palm y (height direction)
    d = p - c
    if yaw:                                   # rotate the solid about the palm y axis (vertical in a top-down grasp)
        cy, sy = np.cos(-yaw), np.sin(-yaw); d = d.copy(); d[:, [0, 2]] = d[:, [0, 2]] @ np.array([[cy, sy], [-sy, cy]])
    h = S / 2
    if shape == "fallen_cylinder":            # axis along palm z (yaw rotates it); curved face up, 45 mm diameter
        return (d[:, 0] ** 2 + d[:, 1] ** 2 < h * h) & (np.abs(d[:, 2]) < h)
    if shape == "cube":
        return (np.abs(d[:, 0]) < h) & (np.abs(d[:, 2]) < h) & (np.abs(d[:, 1]) < h)
    if shape == "cylinder":
        return (d[:, 0] ** 2 + d[:, 2] ** 2 < h * h) & (np.abs(d[:, 1]) < h)
    if shape == "triangle":   # equilateral, altitude S, apex toward +x (flat face towards -x), centroid at c
        a = S; R = 2 * a / 3                      # circumradius
        ang = [0, 2 * np.pi / 3, 4 * np.pi / 3]; V = np.array([[R * np.cos(t), R * np.sin(t)] for t in ang])
        q = d[:, [0, 2]]; ok = np.ones(len(q), bool)
        for i in range(3):                    # inside = same side of every edge as the centroid (origin)
            e = V[(i + 1) % 3] - V[i]; n = np.array([-e[1], e[0]])
            ok &= np.sign((q - V[i]) @ n) == np.sign((-V[i]) @ n)
        return ok & (np.abs(d[:, 1]) < h)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    model = pin.buildModelFromUrdf(str(URDF)); data = model.createData()
    pts = load_pts()
    iq = [model.joints[model.getJointId(j)].idx_q for j in J]
    fid = {k: model.getFrameId(v) for k, v in LINKS.items()}
    lo = np.array([model.lowerPositionLimit[i] for i in iq]); hi = np.array([model.upperPositionLimit[i] for i in iq])
    print("joint limits lo", lo.round(2), "hi", hi.round(2))
    palm_pts = pts["palm"]; pad = palm_pts[:, 1] < palm_pts[:, 1].min() + 0.0005
    print("palm lowest pad y=%.4f  x[%.3f,%.3f] z[%.3f,%.3f]  (palm link origin = wrist side, NOT the contact surface)" % (
        palm_pts[pad, 1].mean(), palm_pts[pad, 0].min(), palm_pts[pad, 0].max(), palm_pts[pad, 2].min(), palm_pts[pad, 2].max()))
    ypad = float(palm_pts[:, 1].min())
    def posed(q):
        qq = pin.neutral(model); qq[iq] = q; pin.framesForwardKinematics(model, data, qq)
        Mp = data.oMf[fid["palm"]]   # express everything in the PALM frame (URDF root is base_link, not the palm)
        return {k: ((pts[k] @ data.oMf[fid[k]].rotation.T + data.oMf[fid[k]].translation) - Mp.translation) @ Mp.rotation
                for k in pts if k != "palm"}
    rep = {}
    OPEN = np.array([0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4])
    TGT = np.array([0.0, 0.92, 1.0, -1.57, -1.2, -1.57, -1.2])     # deepest closing target inside URDF limits
    GROUPS = {"thumb": ([0, 1, 2], ("t0", "t1", "t2")), "middle": ([3, 4], ("m0", "m1")), "index": ([5, 6], ("i0", "i1"))}
    fr = np.linspace(0, 1, 51)
    CASES = [("cube", 0.0), ("cylinder", 0.0)] + [("triangle", np.radians(d)) for d in range(0, 120, 15)] + \
            [("fallen_cylinder", 0.0), ("fallen_cylinder", np.pi / 2)]
    for shape, yaw in CASES:
        rows = []
        for xc in np.arange(0.030, 0.0951, 0.005):
            for zc in np.arange(-0.015, 0.0151, 0.005):
                c = np.array([xc, ypad - S / 2 - 0.0005, zc])        # top face 0.5 mm below the palm pad
                P0 = posed(OPEN)
                if sum(int(inside(shape, p, c, yaw).sum()) for p in P0.values()):
                    continue                                          # open hand would already intersect: no descent
                cover = palm_pts[(palm_pts[:, 1] < ypad + 0.002) & inside(shape, palm_pts * [1, 0, 1] + [0, c[1], 0], c, yaw)]
                first = {}
                for gname, (ji, ks) in GROUPS.items():                # each group closes alone (others open)
                    first[gname] = None
                    for f in fr:
                        q = OPEN.copy(); q[ji] = OPEN[ji] + f * (TGT[ji] - OPEN[ji])
                        P = posed(q)
                        if sum(int(inside(shape, P[k], c, yaw).sum()) for k in ks):
                            first[gname] = round(float(f), 2); break
                rows.append(dict(xc=round(float(xc), 3), zc=round(float(zc), 3), palm_pad_pts_over_top=int(len(cover)), first_contact=first,
                                 all3=all(v is not None for v in first.values())))
        ok = [r for r in rows if r["all3"] and r["palm_pad_pts_over_top"] > 0]
        key = f"{shape}@{np.degrees(yaw):.0f}deg"
        rep[key] = dict(evaluated=len(rows), feasible_palm_and_3finger=len(ok),
                          best=sorted(ok, key=lambda r: (-r["palm_pad_pts_over_top"], max(r["first_contact"].values())))[:3],
                          fail_examples=[r for r in rows if not r["all3"]][:2])
        print(key, "evaluated", len(rows), "feasible", len(ok)); [print("   ", r) for r in rep[key]["best"][:1]]
    rep["palm_pad"] = dict(y=round(ypad, 4), x=[round(float(palm_pts[pad, 0].min()), 4), round(float(palm_pts[pad, 0].max()), 4)],
                           z=[round(float(palm_pts[pad, 2].min()), 4), round(float(palm_pts[pad, 2].max()), 4)])
    if a.out: Path(a.out).write_text(json.dumps(rep, indent=1))
if __name__ == "__main__":
    main()
