"""PERCEPTION_MILESTONES A4: when to abstain, and where camera refinement stops recovering.

    python -m aaltonomous.baselines.classical.loop.vision.a4_sweep --split shift_dev --variant v2 v2_colour_invariant
    python -m aaltonomous.baselines.classical.loop.vision.a4_sweep --pose 1 2 3 4 5 6

Thresholds are applied AFTER detection to the per-object confidences eval_vision stores (color_conf, shape_conf):
colour -> None if color_conf < tc; shape -> None if shape_conf < ts (on top of the fit's own gates). Expected
points per detectable object (sort_language proxy): +2 colour AND shape right, -1 any reported value wrong, 0
abstained or missed; false positives -1 each (they would be sorted somewhere). A colour-only variant of the score
(rung 1 / colour rules) is reported too. Thresholds are picked by the maximum of the curve, not by accuracy.
SYNTHETIC: algorithmic evidence only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import render_split, split_ids
from .eval_vision import evaluate

HERE = Path(__file__).resolve().parent


def points(rows, tc: float, ts: float):
    n_gt = sum(r["n_gt"] for r in rows)
    fp = sum(len(r["fp"]) for r in rows)
    typ = col = wrong_c = unk_c = wrong_s = unk_s = 0
    for r in rows:
        for p in r["pairs"]:
            if not p["detectable"]:
                continue
            c = p["det_color"] if p["color_conf"] >= tc else None
            s = p["det_shape"] if (p["det_shape"] is None or p["shape_conf"] >= ts) else None
            cw, sw = c not in (None, p["gt_color"]), s not in (None, p["gt_shape"])
            wrong_c += cw
            unk_c += c is None
            wrong_s += sw
            unk_s += s is None
            typ += 2 if (c == p["gt_color"] and s == p["gt_shape"]) else (-1 if (cw or sw) else 0)
            col += 2 if c == p["gt_color"] else (-1 if cw else 0)
    return {"type": (typ - fp) / max(n_gt, 1), "colour": (col - fp) / max(n_gt, 1), "wrong_c": wrong_c,
            "unk_c": unk_c, "wrong_s": wrong_s, "unk_s": unk_s}


def sweep(split, variants, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = split_ids(split)
    render_split(split)
    tcs = np.round(np.arange(0.0, 1.0001, 0.025), 3)
    tss = np.round(np.arange(0.0, 0.301, 0.01), 3)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v in variants:
        _, rows = evaluate(v, ids)
        cur_c = [points(rows, t, 0.0) for t in tcs]
        bc = int(np.argmax([c["colour"] for c in cur_c]))
        tc = float(tcs[bc])
        cur_s = [points(rows, tc, t) for t in tss]
        bs = int(np.argmax([c["type"] for c in cur_s]))
        ts = float(tss[bs])
        res[v] = {"tc": tc, "ts": ts, "at_0": points(rows, 0, 0), "at_best": points(rows, tc, ts),
                  "curve_colour": [[float(t), c["colour"], c["wrong_c"], c["unk_c"]] for t, c in zip(tcs, cur_c)],
                  "curve_shape": [[float(t), c["type"], c["wrong_s"], c["unk_s"]] for t, c in zip(tss, cur_s)]}
        ax[0].plot(tcs, [c["colour"] for c in cur_c], label=f"{v} (best {tc:.3f})")
        ax[1].plot(tss, [c["type"] for c in cur_s], label=f"{v} (best {ts:.2f})")
        print(f"{v:<22} colour min_conf={tc:.3f}  shape min margin={ts:.2f}  "
              f"pts/obj type {res[v]['at_0']['type']:+.3f} -> {res[v]['at_best']['type']:+.3f}, "
              f"colour {res[v]['at_0']['colour']:+.3f} -> {res[v]['at_best']['colour']:+.3f}; "
              f"colour wrong {res[v]['at_0']['wrong_c']} -> {res[v]['at_best']['wrong_c']} "
              f"(unk {res[v]['at_best']['unk_c']}), shape wrong {res[v]['at_0']['wrong_s']} -> "
              f"{res[v]['at_best']['wrong_s']} (unk {res[v]['at_best']['unk_s']})")
    ax[0].set(xlabel="colour min_conf (vote share)", ylabel="expected points / object (colour rules)",
              title=f"A4 colour abstain, {split} (synthetic)")
    ax[1].set(xlabel="extra shape margin", ylabel="expected points / object (colour + shape rules)",
              title="A4 shape abstain at the chosen colour threshold")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    png = Path(out).with_suffix(".png")
    fig.savefig(png, dpi=110)
    Path(out).write_text(json.dumps(res, indent=1))
    print("plot ->", png, " json ->", out)
    return res


def pose(levels, variant="v2"):
    print(f"camera pose error sweep ({variant}; train family otherwise; 30 band frames per level)")
    out = {}
    for k in levels:
        name = f"pose_{k}"
        render_split(name, 16)
        s, rows = evaluate(variant, split_ids(name))
        a = s["all"]
        out[k] = a
        print(f"  +-{k:>3} cm / deg: recall {100 * a['recall']:5.1f} %  FP {a['fp']:>2}  colour wrong "
              f"{a['color_wrong']}  shape wrong {a['shape_wrong']} unk {a['shape_unknown']}  xy med/p95/max "
              f"{a['xy_med_mm']:.1f}/{a['xy_p95_mm']:.1f}/{a['xy_max_mm']:.1f} mm  yaw p95 "
              f"{(a['yaw_p95_deg'] or float('nan')):.1f} deg")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="shift_dev")
    ap.add_argument("--variant", nargs="+", default=["v2"])
    ap.add_argument("--pose", nargs="*", default=None, help="camera pose error levels (cm / deg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.pose is not None:
        res = pose(a.pose, a.variant[0])
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    sweep(a.split, a.variant, a.out or str(HERE / "results" / f"a4_{a.split}.json"))


if __name__ == "__main__":
    main()
