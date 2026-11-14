"""Compare perception approaches on REAL head frames with hand labels (no depth, no metric ground truth).

    python -m aaltonomous.baselines.classical.loop.vision.eval_real --variant legacy v2 detector ...

Labels: real/drop2_labels.json (colour, shape, pixel centre; see its header). A detection matches a label when its
blob pixel (Detection.uv) is within 40 px (Hungarian). Per variant: recall over labels not cut by the image edge;
FP = detections matching no label (robot hands, bins, glare, the orange tape); colour right / unknown / wrong with
orange->red, purple->blue; shape right / unknown / wrong over labels whose shape is known; points per labelled object
(+2 colour and shape right, -1 any wrong value, 0 otherwise; FP -1). Evidence label: `real` (but no depth stream,
camera pose not the calibrated one, objects not the competition colours).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .eval_vision import make_perception

HERE = Path(__file__).resolve().parent
CMAP = {"orange": "red", "purple": "blue", "yellow": "yellow"}
GATE = 40.0


def run(variant, labels, root):
    from scipy.optimize import linear_sum_assignment
    per = make_perception(variant)
    st = dict(n=0, found=0, fp=0, c_ok=0, c_unk=0, c_wrong=0, s_n=0, s_ok=0, s_unk=0, s_wrong=0, pts=0, ms=[])
    rows = []
    for fid, labs in labels["frames"].items():
        z = np.load(Path(root) / f"{fid}.npz")
        rgb, depth = z["rgb"], z["depth"]
        per.reset_calibration()
        t = time.perf_counter()
        dets = per.detect(rgb, depth)
        st["ms"].append((time.perf_counter() - t) * 1000)
        D = np.array([[np.hypot(d.uv[0] - l[2], d.uv[1] - l[3]) for l in labs] for d in dets]) if dets else np.zeros((0, len(labs)))
        pairs = []
        if len(dets):
            r, c = linear_sum_assignment(np.where(D <= GATE, D, 1e6))
            pairs = [(i, j) for i, j in zip(r, c) if D[i, j] <= GATE]
        md, ml = {i for i, _ in pairs}, {j for _, j in pairs}
        fp = len(dets) - len(md)
        st["fp"] += fp
        st["pts"] -= fp
        for j, l in enumerate(labs):
            if l[4] == "trunc":
                continue
            st["n"] += 1
            if j not in ml:
                continue
            st["found"] += 1
            d = dets[next(i for i, jj in pairs if jj == j)]
            gc, gs = CMAP[l[0]], l[1]
            cw = d.color is not None and d.color != gc
            if cw:
                st.setdefault("confusions", {}).setdefault(f"{l[0]}->{d.color}", 0)
                st["confusions"][f"{l[0]}->{d.color}"] += 1
            st["c_ok"] += d.color == gc
            st["c_unk"] += d.color is None
            st["c_wrong"] += cw
            sw = False
            if gs is not None:
                st["s_n"] += 1
                st["s_ok"] += d.shape == gs
                st["s_unk"] += d.shape is None
                sw = d.shape is not None and d.shape != gs
                st["s_wrong"] += sw
            ok_s = d.shape == gs if gs is not None else True
            st["pts"] += 2 if (d.color == gc and ok_s and gs is not None) else (-1 if (cw or sw) else 0)
        rows.append({"id": fid, "dets": [repr(d) for d in dets], "fp": fp})
    return st, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs="+", default=["legacy", "v2"])
    ap.add_argument("--labels", default=str(HERE / "real" / "drop2_labels.json"))
    ap.add_argument("--root", default="/workspace/data/real_drops/frames/g1_junction_drop_2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    labels = json.loads(Path(a.labels).read_text())
    res = {}
    print(f"REAL frames: {len(labels['frames'])} labelled (no depth; orange->red, purple->blue)")
    for v in a.variant:
        st, rows = run(v, labels, a.root)
        res[v] = {"stats": {k: (float(np.median(x)) if k == "ms" else x) for k, x in st.items()}, "rows": rows}
        n = max(st["n"], 1)
        print(f"{v:<22} recall {st['found']}/{st['n']} ({100 * st['found'] / n:4.1f} %)  FP {st['fp']:>3}  "
              f"colour ok/unk/wrong {st['c_ok']}/{st['c_unk']}/{st['c_wrong']}  shape ok/unk/wrong "
              f"{st['s_ok']}/{st['s_unk']}/{st['s_wrong']} (of {st['s_n']} known)  pts/obj {st['pts'] / n:+.2f}  "
              f"ms med {np.median(st['ms']):.0f}  colour errors {st.get('confusions', {})}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
