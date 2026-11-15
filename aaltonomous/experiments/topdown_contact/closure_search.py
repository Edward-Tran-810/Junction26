"""Random search (URDF joint limits) for palm-pad-on-top + 3-finger closure WITH the table plane. Object centre at palm
x = xc (palm pad must be the first link to touch the top: xc >= 0.060, contact_link_check), top 0.5 mm below the pad.
Per finger group: preshape p and target g sampled uniformly in the joint limits; valid if (a) p is free of object and
table (+3 mm), (b) the straight joint path p -> g touches the object before the table (+1 mm). Reports the best by the
contact height on the side faces (lower = more wrap) and writes closure_report.json.
Run: python -m aaltonomous.experiments.topdown_contact.closure_search [--n 4000]"""
import argparse, json, numpy as np, pinocchio as pin
from pathlib import Path
from aaltonomous.experiments.topdown_contact import geometry_check as G

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4000); ap.add_argument("--gaps_mm", default="0.5")
    ap.add_argument("--xcs", default="0.060,0.065,0.070"); ap.add_argument("--shapes", default="cube@0,cylinder@0,triangle@0,triangle@180,fallen_cylinder@90")
    ap.add_argument("--out", default="closure_report.json"); a = ap.parse_args()
    model = pin.buildModelFromUrdf(str(G.URDF)); data = model.createData()
    iq = [model.joints[model.getJointId(j)].idx_q for j in G.J]
    fid = {k: model.getFrameId(v) for k, v in G.LINKS.items()}
    lo = np.array([model.lowerPositionLimit[i] for i in iq]); hi = np.array([model.upperPositionLimit[i] for i in iq])
    pts = G.load_pts(400); ypad = float(pts["palm"][:, 1].min())
    GR = {"thumb": ([0, 1, 2], ("t0", "t1", "t2")), "middle": ([3, 4], ("m0", "m1")), "index": ([5, 6], ("i0", "i1"))}
    def posed(q, keys):
        qq = pin.neutral(model); qq[iq] = q; pin.framesForwardKinematics(model, data, qq); Mp = data.oMf[fid["palm"]]
        return np.vstack([((pts[k] @ data.oMf[fid[k]].rotation.T + data.oMf[fid[k]].translation) - Mp.translation) @ Mp.rotation for k in keys])
    rng = np.random.default_rng(0); rep = {}
    for sy in a.shapes.split(","):
      shape, yd = sy.split("@"); yaw = np.radians(float(yd))
      for gap in [float(v) * 1e-3 for v in a.gaps_mm.split(",")]:
        ytab = ypad - gap - G.S
        key = f"{shape}@{yd}|gap{gap * 1e3:.1f}mm"; rep[key] = {}
        for xc in [float(v) for v in a.xcs.split(",")]:
            c = np.array([xc, ypad - G.S / 2 - gap, 0.0]); res = {}
            for gname, (ji, ks) in GR.items():
                best = None
                for _ in range(a.n):
                    p = rng.uniform(lo[ji], hi[ji]); g = rng.uniform(lo[ji], hi[ji])
                    if gname == "thumb": p[0] = g[0] = rng.choice([0.0, -0.5, 0.5])     # thumb_0 swing kept constant on a path
                    q = np.zeros(7); q[ji] = p; P = posed(q, ks)
                    if G.inside(shape, P, c, yaw).any() or (P[:, 1] < ytab + 0.003).any(): continue
                    for f in np.linspace(0, 1, 31)[1:]:
                        q[ji] = p + f * (g - p); P = posed(q, ks)
                        if (P[:, 1] < ytab + 0.001).any(): break
                        hit = G.inside(shape, P, c, yaw)
                        if hit.any():
                            h = float((P[hit, 1].mean() - ytab) * 1e3); xs = float(P[hit, 0].mean() - xc)
                            cand = dict(pre=p.round(3).tolist(), target=g.round(3).tolist(), q_contact=q[ji].round(3).tolist(),
                                        contact_h_above_table_mm=round(h, 1), contact_dx_mm=round(xs * 1e3, 1), f=round(float(f), 2))
                            if best is None or h < best["contact_h_above_table_mm"]: best = cand
                            break
                res[gname] = best
            # opposition: thumb contact on the -x side and fingers on the +x side of the centre (a real squeeze, not a press)
            opp = all(res[g] for g in GR) and res["thumb"]["contact_dx_mm"] < -5 and min(res["middle"]["contact_dx_mm"], res["index"]["contact_dx_mm"]) > 5
            rep[key][f"{xc:.3f}"] = dict(groups=res, all3=all(res[g] for g in GR), opposed=bool(opp))
            print(key, xc, "all3" if rep[key][f"{xc:.3f}"]["all3"] else "-", "opposed" if opp else "NOT opposed",
                  {g: (r and (r["contact_h_above_table_mm"], r["contact_dx_mm"])) for g, r in res.items()}, flush=True)
    rep["_meta"] = dict(palm_pad_y=round(ypad, 4), samples_per_group=a.n,
                        note="contact_h = mean height of the contact above the table (45 mm = top edge); contact_dx = x offset from object centre")
    (Path(__file__).parent / a.out).write_text(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
