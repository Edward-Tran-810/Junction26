"""Search a Dex3 preshape + closing targets for palm-on-top grasps WITH the table (offline, URDF collision geometry).
Palm pad on the object top (0.5 mm gap); the table is the plane 45 mm below the top. Constraints:
  preshape : no hand point inside the object or below table + 3 mm (the hand can descend onto the object)
  closing  : each group (thumb / middle / index) moves preshape -> target alone; it must touch the object before it
             reaches the table plane (+1 mm) and its first contact must happen before the target.
Writes preshape_report.json next to this file. Run: python -m aaltonomous.experiments.topdown_contact.preshape_search"""
import itertools, json, numpy as np, pinocchio as pin
from pathlib import Path
from aaltonomous.experiments.topdown_contact import geometry_check as G

def main():
    model = pin.buildModelFromUrdf(str(G.URDF)); data = model.createData()
    iq = [model.joints[model.getJointId(j)].idx_q for j in G.J]
    fid = {k: model.getFrameId(v) for k, v in G.LINKS.items()}
    pts = G.load_pts(600)
    ypad = float(pts["palm"][:, 1].min())
    def posed(q, keys):
        qq = pin.neutral(model); qq[iq] = q; pin.framesForwardKinematics(model, data, qq); Mp = data.oMf[fid["palm"]]
        return {k: ((pts[k] @ data.oMf[fid[k]].rotation.T + data.oMf[fid[k]].translation) - Mp.translation) @ Mp.rotation for k in keys}
    GR = {"thumb": ([0, 1, 2], ("t0", "t1", "t2")), "middle": ([3, 4], ("m0", "m1")), "index": ([5, 6], ("i0", "i1"))}
    ytab = ypad - 0.0005 - G.S
    cases = {"cube": ("cube", 0.0, [0.055, 0.060, 0.065]), "cylinder": ("cylinder", 0.0, [0.055, 0.060, 0.065]),
             "triangle": ("triangle", 0.0, [0.050, 0.055, 0.060]), "fallen_cylinder": ("fallen_cylinder", np.pi / 2, [0.055, 0.060, 0.065])}
    # candidate per-group preshapes / targets (rad), inside URDF limits
    th_pre = [(t0, t1, t2) for t0 in (0.0,) for t1 in (-0.72, -0.6, -0.4, -0.2) for t2 in (0.0, 0.4, 0.8, 1.2, 1.5, 1.75)]
    th_tgt = [(0.0, t1, t2) for t1 in (0.4, 0.7, 0.92) for t2 in (0.0, 0.3, 0.6, 0.9, 1.2)]
    fi_pre = [(a, b) for a in (-0.1, -0.3, -0.5) for b in (-0.1, -0.3, -0.5, -0.8)]
    fi_tgt = [(a, b) for a in (-1.2, -1.4, -1.57) for b in (-0.6, -0.9, -1.2, -1.5)]
    rep = {}
    for name, (shape, yaw, xcs) in cases.items():
        best = None
        for xc in xcs:
            c = np.array([xc, ypad - G.S / 2 - 0.0005, 0.0])
            def bad(P, m=0.003):     # inside object or below table+m
                return sum(int(G.inside(shape, p, c, yaw).sum()) + int((p[:, 1] < ytab + m).sum()) for p in P.values())
            res = {}
            for gname, (ji, ks) in GR.items():
                pres = th_pre if gname == "thumb" else fi_pre
                tgts = th_tgt if gname == "thumb" else fi_tgt
                opts = []
                for pre in pres:
                    q = np.zeros(7); q[ji] = pre
                    if bad(posed(q, ks)):
                        continue
                    for tg in tgts:
                        first = None
                        for f in np.linspace(0, 1, 41)[1:]:
                            q2 = np.zeros(7); q2[ji] = np.array(pre) + f * (np.array(tg) - np.array(pre))
                            P = posed(q2, ks)
                            hit = sum(int(G.inside(shape, p, c, yaw).sum()) for p in P.values())
                            tab = sum(int((p[:, 1] < ytab + 0.001).sum()) for p in P.values())
                            if tab: break
                            if hit:
                                first = (float(f), q2[ji].round(3).tolist()); break
                        if first:
                            # contact point height above the table: lower contacts on the side face hold better
                            P = posed(q2, ks); allp = np.vstack(list(P.values()))
                            hp = allp[G.inside(shape, allp, c, yaw)]
                            opts.append(dict(pre=list(pre), target=list(tg), contact_f=round(first[0], 3), q_contact=first[1],
                                             contact_h_above_table_mm=round(float((hp[:, 1].mean() - ytab) * 1e3), 1)))
                # prefer contact in the lower half of the object, then early contact (less sweep)
                opts.sort(key=lambda o: (abs(o["contact_h_above_table_mm"] - 15), o["contact_f"]))
                res[gname] = opts[:2]
            ok = all(res[g] for g in GR)
            score = sum(res[g][0]["contact_h_above_table_mm"] for g in GR) if ok else None
            if ok and (best is None or abs(xc - 0.06) < abs(best[0] - 0.06)):
                best = (xc, res)
            print(name, "xc", xc, "feasible" if ok else "infeasible", {g: (res[g][0] if res[g] else None) for g in GR}, flush=True)
        rep[name] = None if best is None else {"object_centre_in_palm": [best[0], round(ypad - G.S / 2 - 0.0005, 4), 0.0], "groups": best[1]}
    rep["table_y_in_palm"] = round(ytab, 4)
    (Path(__file__).parent / "preshape_report.json").write_text(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
