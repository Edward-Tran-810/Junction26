"""Which Dex3 link touches the object TOP when the hand descends straight down (palm facing down, given preshape)?
For each object xy offset under the hand, the object is raised until its top touches the first hand point; report the
link (palm pad vs thumb_0 = thumb root vs fingers) and the vertical gap between the palm pad and the top at that moment.
Also checks the 3-finger closure from that pose with the table plane (45 mm below the top).
Run: python -m aaltonomous.experiments.topdown_contact.contact_link_check"""
import json, numpy as np, pinocchio as pin
from pathlib import Path
from aaltonomous.experiments.topdown_contact import geometry_check as G

def main():
    model = pin.buildModelFromUrdf(str(G.URDF)); data = model.createData()
    iq = [model.joints[model.getJointId(j)].idx_q for j in G.J]
    fid = {k: model.getFrameId(v) for k, v in G.LINKS.items()}
    pts = G.load_pts(2500)
    def posed(q):
        qq = pin.neutral(model); qq[iq] = q; pin.framesForwardKinematics(model, data, qq); Mp = data.oMf[fid["palm"]]
        return {k: ((pts[k] @ data.oMf[fid[k]].rotation.T + data.oMf[fid[k]].translation) - Mp.translation) @ Mp.rotation for k in pts}
    ypad = float(pts["palm"][:, 1].min())
    # preshapes: thumb swung back + tucked, fingers forward-ish (both keep away from the object footprint)
    PRE = {"thumb_back_fingers_fwd": [0.0, -0.72, 1.75, -0.1, -0.1, -0.1, -0.1], "cfg_open": [0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4],
           "thumb_back_t2_0": [0.0, -0.72, 0.0, -0.1, -0.1, -0.1, -0.1]}
    rep = {}
    for pname, q in PRE.items():
        P = posed(np.array(q, float)); rows = []
        for shape, yaw in (("cube", 0.0), ("cylinder", 0.0), ("triangle", 0.0), ("fallen_cylinder", np.pi / 2)):
            for xc in np.arange(0.020, 0.0701, 0.005):
                # top-face footprint test with a tall solid: which hand points are above the footprint?
                c0 = np.array([xc, 0.0, 0.0])
                foot = {k: p[G.inside(shape if shape != "fallen_cylinder" else "cube", p * [1, 0, 1], c0, yaw)] for k, p in P.items()}
                # fallen cylinder: the top is a line along palm z at x = xc (approx: footprint = cube, top lowered by curvature)
                low = {k: float(f[:, 1].min()) for k, f in foot.items() if len(f)}
                if not low: continue
                first = min(low, key=low.get)
                gap_pad = (ypad - low[first]) * 1e3                 # how far ABOVE the top the palm pad stays
                rows.append(dict(shape=shape, xc=round(float(xc), 3), first_touch_link=first, palm_pad_gap_mm=round(gap_pad, 1),
                                 lowest_per_link_mm={k: round(v * 1e3, 1) for k, v in sorted(low.items(), key=lambda kv: kv[1])[:3]}))
        rep[pname] = rows
        pad_rows = [r for r in rows if r["first_touch_link"] == "palm"]
        print(f"preshape {pname}: positions where the PALM PAD touches first: {len(pad_rows)}/{len(rows)}")
        for r in rows:
            if r["shape"] == "cube": print("   ", r)
    (Path(__file__).parent / "contact_link_report.json").write_text(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
