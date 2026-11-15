"""Check ONE Dex3 preshape -> goal closing path for the near-contact top-down grasp (offline, URDF collision geometry, table
plane included). For each shape / object x under the palm / palm gap: which group touches the object first along the path
(each group alone and all together), contact height above the table, and table / object penetration of the preshape.
Run: python -m aaltonomous.experiments.topdown_contact.posture_check --pre ... --goal ..."""
import argparse, json, numpy as np, pinocchio as pin
from aaltonomous.experiments.topdown_contact import geometry_check as G

GR = {"thumb": ([0, 1, 2], ("t0", "t1", "t2")), "middle": ([3, 4], ("m0", "m1")), "index": ([5, 6], ("i0", "i1"))}

class Hand:
    def __init__(self, n=800):
        self.model = pin.buildModelFromUrdf(str(G.URDF)); self.data = self.model.createData()
        self.iq = [self.model.joints[self.model.getJointId(j)].idx_q for j in G.J]
        self.fid = {k: self.model.getFrameId(v) for k, v in G.LINKS.items()}
        self.pts = G.load_pts(n); self.ypad = float(self.pts["palm"][:, 1].min())
    def posed(self, q, keys):
        qq = pin.neutral(self.model); qq[self.iq] = q; pin.framesForwardKinematics(self.model, self.data, qq)
        Mp = self.data.oMf[self.fid["palm"]]
        return {k: ((self.pts[k] @ self.data.oMf[self.fid[k]].rotation.T + self.data.oMf[self.fid[k]].translation) - Mp.translation) @ Mp.rotation for k in keys}

def check(h, pre, goal, shape, yaw, xc, gap, steps=60):
    c = np.array([xc, h.ypad - G.S / 2 - gap, 0.0]); ytab = h.ypad - gap - G.S
    P = h.posed(pre, [k for k in G.LINKS if k != "palm"]); allp = np.vstack(list(P.values()))
    out = {"pre_obj_pen": int(G.inside(shape, allp, c, yaw).sum()), "pre_table_mm": round(float((allp[:, 1].min() - ytab) * 1e3), 1)}
    for gname, (ji, ks) in GR.items():
        r = None
        for f in np.linspace(0, 1, steps + 1)[1:]:
            q = np.array(pre, float); q[ji] = np.array(pre)[ji] + f * (np.array(goal)[ji] - np.array(pre)[ji])
            Pk = np.vstack(list(h.posed(q, ks).values()))
            tab = float((Pk[:, 1].min() - ytab) * 1e3)
            hit = G.inside(shape, Pk, c, yaw)
            if tab < 1.0: r = {"result": "TABLE", "f": round(float(f), 3)}; break
            if hit.any():
                r = {"result": "contact", "f": round(float(f), 3), "h_mm": round(float((Pk[hit, 1].mean() - ytab) * 1e3), 1),
                     "dx_mm": round(float((Pk[hit, 0].mean() - xc) * 1e3), 1), "q": q[ji].round(3).tolist(), "table_clear_mm": round(tab, 1)}
                break
        out[gname] = r or {"result": "none"}
    out["ok"] = (out["pre_obj_pen"] == 0 and out["pre_table_mm"] >= 3 and all(out[g]["result"] == "contact" for g in GR)
                 and out["thumb"]["dx_mm"] < -5 and min(out["middle"]["dx_mm"], out["index"]["dx_mm"]) > 5)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True); ap.add_argument("--goal", required=True)
    ap.add_argument("--shapes", default="cube@0,cylinder@0,triangle@180"); ap.add_argument("--xcs", default="0.07,0.075,0.08,0.085")
    ap.add_argument("--gaps_mm", default="15,20,25"); a = ap.parse_args()
    h = Hand(); pre = [float(v) for v in a.pre.split(",")]; goal = [float(v) for v in a.goal.split(",")]
    for sy in a.shapes.split(","):
        s, yd = sy.split("@")
        for gap in a.gaps_mm.split(","):
            for xc in a.xcs.split(","):
                r = check(h, pre, goal, s, np.radians(float(yd)), float(xc), float(gap) * 1e-3)
                print(f"{sy:13s} gap {gap}mm xc {xc}: {'OK ' if r['ok'] else '-- '}", json.dumps({k: v for k, v in r.items() if k != 'ok'}), flush=True)
