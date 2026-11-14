"""Score a perception variant on a frozen frame split (synthetic now; recorded sim / real frames later).

    python -m aaltonomous.baselines.classical.loop.vision.eval_vision --split dev --variant v2
    python -m aaltonomous.baselines.classical.loop.vision.eval_vision --split fresh_01 --variant legacy v2 \
        --out loop/vision/results/round_01_fresh.json
    python -m aaltonomous.baselines.classical.loop.vision.eval_vision --split dev --variant v2 --overlays 10

Metric definitions (also in loop/STATE.md; they apply per split and are reported per family):
    detectable GT    an object fully inside the image (not `truncated` by the image border), upright (recorded
                     frames: not `tipped` > 10 deg), with
                     visible_px >= 0.5 * unoccluded_px (at most half hidden by another object or an occluder) and
                     visible_px >= 200. Others are "hard": not counted as misses, and a detection on them is not a
                     false positive.
    match            Hungarian assignment on the pelvis-xy distance between Detection.top and the GT top-face
                     centre, gated at MATCH_GATE_M (30 mm, = 2/3 of the object size; objects are >= 65 mm apart).
                     Colour is NOT required to match (colour accuracy is reported separately).
    recall           matched detectable GT / detectable GT.
    false positive   a detection matched to no GT object. fp_rate = FPs / all detections; also FPs per frame.
    accuracies       colour, shape, position and yaw are computed over matched DETECTABLE objects (the population of
                     recall). Matches on "hard" (more than half hidden) objects are reported separately:
                     hard_matched, and hard_shape_wrong (a wrong shape there would still be a real error).
    colour accuracy  matched detections with the right colour / matched.
    shape accuracy   matched detections with the right shape / matched. shape None ("not sure") is counted as
                     NOT correct, and reported separately as `shape_unknown` (it is safe: the policy then only
                     sorts the colour if every shape of it goes to the same bin), `shape_wrong` is the unsafe kind.
    position error   pelvis-xy distance (mm) of matched detections: median, 95th percentile, max. z error separate.
    yaw error        cubes and triangles only, matched detections: |wrap(yaw_det - yaw_gt)| modulo the GT shape's
                     symmetry (90 deg cube, 120 deg triangle), in deg; a missing yaw is counted in `yaw_missing`.
    latency          wall time of Perception.detect() per frame, single thread (cv2 + BLAS limited to 1 thread),
                     median and 95th percentile, after one warm-up frame. "cold" = first frame of an episode
                     (camera refinement estimated: every frame here is its own scene); "warm" = the same frame with
                     the cached camera correction (steady state inside an episode; refine_cache in config.yaml).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .dataset import load_frame, render_split, split_ids  # noqa: E402

MATCH_GATE_M = 0.030
SYMMETRY_DEG = {"cube": 90.0, "triangle": 120.0}
HERE = Path(__file__).resolve().parent

# Perception variants: config overrides on top of config.yaml (dotted keys, see config.load_config)
VARIANTS = {
    "legacy": {"perception.pipeline": "legacy"},
    "v2": {"perception.pipeline": "v2"},
    "v2_polygon": {"perception.pipeline": "v2", "perception.shape.method": "polygon"},
    "v2_render": {"perception.pipeline": "v2", "perception.shape.method": "render"},
    "v2_combined": {"perception.pipeline": "v2", "perception.shape.method": "combined"},
    "v2_nodepth": {"perception.pipeline": "v2", "_drop_depth": True},
    # PERCEPTION_MILESTONES A3 candidates (same Detection output, same evaluator)
    "v2_colour_invariant": {"perception.pipeline": "v2", "perception.colour.mode": "wb_proto"},
    "cloud": {"perception.pipeline": "v2", "perception.colour.keep_uncoloured": True},
    "detector": {"perception.pipeline": "v2", "perception.segmenter": "detector"},
    "open_vocab": {"perception.pipeline": "v2", "perception.segmenter": "open_vocab"},
    "vlm_verify": {"perception.pipeline": "v2", "perception.vlm_verify": True},
    "detector_hue": {"perception.pipeline": "v2", "perception.segmenter": "detector",
                     "perception.colour_from": "hue_cluster"},
    "detector_hue_real": {"perception.pipeline": "v2", "perception.segmenter": "detector",
                          "perception.colour_from": "hue_cluster", "perception.hue.prototypes": "real_drop2"},
    "detector_hue_agree": {"perception.pipeline": "v2", "perception.segmenter": "detector",
                           "perception.colour_from": "hue_cluster", "perception.hue.agree_with_model": True},
    "detector_hue_wb": {"perception.pipeline": "v2", "perception.segmenter": "detector",
                        "perception.colour_from": "hue_cluster", "perception.hue.white_balance": True},
    "detector_hue_wb_agree": {"perception.pipeline": "v2", "perception.segmenter": "detector",
                              "perception.colour_from": "hue_cluster", "perception.hue.white_balance": True,
                              "perception.hue.agree_with_model": True},
    "open_vocab_drop_words": {"perception.pipeline": "v2", "perception.segmenter": "open_vocab",
                              "perception.open_vocab_words": {"red": "orange", "blue": "purple"}},
    "v2_margin": {"perception.pipeline": "v2", "perception.colour.conf": "share_x_margin"},
    "v2_colour_invariant_margin": {"perception.pipeline": "v2", "perception.colour.mode": "wb_proto",
                                   "perception.colour.conf": "share_x_margin"},
    "cloud_colour_invariant": {"perception.pipeline": "v2", "perception.colour.keep_uncoloured": True,
                               "perception.colour.mode": "wb_proto"},
    "legacy_nodepth": {"perception.pipeline": "legacy", "_drop_depth": True},
}


def make_perception(variant: str, extra: dict | None = None):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception import Perception
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    ov = {k: v for k, v in VARIANTS[variant].items() if not k.startswith("_")}
    ov.update(extra or {})
    return Perception(load_config(None, ov), Scene.load(), Calibration.load())


def wrap_deg(a: float, period: float) -> float:
    """Smallest |a| modulo `period` (deg), in [0, period/2]."""
    a = abs(a) % period
    return min(a, period - a)


def match(dets, gts):
    """Hungarian matching on pelvis-xy distance with gate. Returns list of (det index, gt index, distance m)."""
    from scipy.optimize import linear_sum_assignment
    if not dets or not gts:
        return []
    D = np.array([[np.hypot(*(np.asarray(d.top[:2]) - np.asarray(g["top"][:2]))) if np.all(np.isfinite(d.top))
                   else 1e3 for g in gts] for d in dets])
    r, c = linear_sum_assignment(np.where(D <= MATCH_GATE_M, D, 1e3))
    return [(i, j, D[i, j]) for i, j in zip(r, c) if D[i, j] <= MATCH_GATE_M]


def detectable(g) -> bool:
    return (not g.get("truncated", False) and not g.get("tipped", False) and g["visible_px"] >= 200
            and g["visible_px"] >= 0.5 * max(g["unoccluded_px"], 1))


def evaluate(variant: str, ids: list[str], extra: dict | None = None, keep_rows: bool = False):
    """Run one variant over frames; returns (summary dict, per-frame rows)."""
    import cv2
    cv2.setNumThreads(1)
    per = make_perception(variant, extra)
    drop_depth = VARIANTS[variant].get("_drop_depth", False)
    rows = []
    warm = False
    for fid in ids:
        rgb, depth, gt = load_frame(fid)
        if drop_depth:
            depth = np.zeros_like(depth)
        if not warm:
            per.detect(rgb, depth)
            warm = True
        per.reset_calibration()                 # every frame is a new scene (own camera error): cold start
        t0 = time.perf_counter()
        dets = per.detect(rgb, depth)
        ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()                # same frame again with the cached camera correction (steady state)
        per.detect(rgb, depth)
        ms_warm = (time.perf_counter() - t0) * 1000
        gts = gt["objects"]
        m = match(dets, gts)
        md, mg = {i for i, _, _ in m}, {j for _, j, _ in m}
        row = {"id": fid, "family": gt["family"], "mode": gt["layout_mode"], "ms": ms, "ms_warm": ms_warm,
               "n_det": len(dets),
               "n_gt": sum(detectable(g) for g in gts),
               "miss": [j for j, g in enumerate(gts) if detectable(g) and j not in mg],
               "fp": [i for i in range(len(dets)) if i not in md], "pairs": []}
        for i, j, dist in m:
            d, g = dets[i], gts[j]
            yaw_err = None
            if g["shape"] in SYMMETRY_DEG and d.yaw is not None:
                yaw_err = wrap_deg(np.degrees(d.yaw - g["yaw"]), SYMMETRY_DEG[g["shape"]])
            row["pairs"].append({"gt": j, "color_ok": d.color == g["color"], "gt_color": g["color"],
                                 "det_color": d.color, "gt_shape": g["shape"], "det_shape": getattr(d, "shape", None),
                                 "xy_mm": dist * 1000, "z_mm": (d.top[2] - g["top"][2]) * 1000,
                                 "yaw_err": yaw_err, "detectable": detectable(g), "method": d.method,
                                 "color_conf": float(getattr(d, "color_conf", 1.0)),
                                 "shape_conf": float(getattr(d, "shape_conf", 0.0))})
        row["fp_desc"] = [repr(dets[i]) for i in row["fp"]]
        row["close_pair"] = gt.get("close_pair")
        row["occl_target"] = gt.get("occl_target")
        row["hidden"] = [1 - g["visible_px"] / max(g["unoccluded_px"], 1) for g in gts]
        rows.append(row)
    return summarize(rows), rows


def summarize(rows, by_family=True):
    def s(rs):
        n_gt = sum(r["n_gt"] for r in rs)
        n_det = sum(r["n_det"] for r in rs)
        allpairs = [p for r in rs for p in r["pairs"]]
        dp = [p for p in allpairs if p["detectable"]]
        pairs = dp                     # accuracy metrics over DETECTABLE objects (same population as recall)
        hard = [p for p in allpairs if not p["detectable"]]
        fp = sum(len(r["fp"]) for r in rs)
        xy = np.array([p["xy_mm"] for p in pairs]) if pairs else np.zeros(0)
        z = np.array([abs(p["z_mm"]) for p in pairs]) if pairs else np.zeros(0)
        yaw_pairs = [p for p in pairs if p["gt_shape"] in SYMMETRY_DEG]
        ye = np.array([p["yaw_err"] for p in yaw_pairs if p["yaw_err"] is not None])
        ms = np.array([r["ms"] for r in rs])
        msw = np.array([r.get("ms_warm", np.nan) for r in rs])
        shape_ok = sum(p["det_shape"] == p["gt_shape"] for p in pairs)
        col_unk = sum(p["det_color"] is None for p in pairs)
        col_wrong = sum(p["det_color"] not in (None, p["gt_color"]) for p in pairs)
        # expected sort_language points per detectable object (proxy): +2 colour AND shape right, -1 any reported
        # value wrong, 0 abstained or missed; colour-only = the same with the shape ignored (rung 1 / colour rules)
        sc_type = sum(2 if (p["det_color"] == p["gt_color"] and p["det_shape"] == p["gt_shape"]) else
                      (-1 if (p["det_color"] not in (None, p["gt_color"]) or p["det_shape"] not in (None, p["gt_shape"]))
                       else 0) for p in pairs)
        sc_col = sum(2 if p["det_color"] == p["gt_color"] else (-1 if p["det_color"] is not None else 0) for p in pairs)
        sc_col -= fp                   # a false positive that gets sorted lands in a bin it may not belong to
        shape_unknown = sum(p["det_shape"] is None for p in pairs)
        out = {"frames": len(rs), "gt": n_gt, "det": n_det, "matched": len(pairs),
               "recall": len(dp) / max(n_gt, 1), "fp": fp, "fp_rate": fp / max(n_det, 1),
               "fp_per_frame": fp / max(len(rs), 1),
               "color_acc": sum(p["color_ok"] for p in pairs) / max(len(pairs), 1),
               "shape_acc": shape_ok / max(len(pairs), 1), "shape_unknown": shape_unknown,
               "shape_wrong": len(pairs) - shape_ok - shape_unknown,
               "color_unknown": col_unk, "color_wrong": col_wrong,
               "score_type_per_obj": sc_type / max(n_gt, 1), "score_colour_per_obj": sc_col / max(n_gt, 1),
               "xy_med_mm": float(np.median(xy)) if len(xy) else None,
               "xy_p95_mm": float(np.percentile(xy, 95)) if len(xy) else None,
               "xy_max_mm": float(xy.max()) if len(xy) else None,
               "z_med_mm": float(np.median(z)) if len(z) else None,
               "yaw_n": len(yaw_pairs), "yaw_missing": len(yaw_pairs) - len(ye),
               "yaw_med_deg": float(np.median(ye)) if len(ye) else None,
               "yaw_p95_deg": float(np.percentile(ye, 95)) if len(ye) else None,
               "hard_matched": len(hard), "hard_shape_wrong": sum(p["det_shape"] not in (None, p["gt_shape"])
                                                                  for p in hard),
               "ms_med": float(np.median(ms)) if len(ms) else None,
               "ms_p95": float(np.percentile(ms, 95)) if len(ms) else None,
               "ms_warm_med": float(np.nanmedian(msw)) if np.isfinite(msw).any() else None,
               "ms_warm_p95": float(np.nanpercentile(msw, 95)) if np.isfinite(msw).any() else None}
        per_shape = {}
        for sh in ("cube", "cylinder", "triangle"):
            ps = [p for p in pairs if p["gt_shape"] == sh]
            if ps:
                per_shape[sh] = {"n": len(ps), "shape_ok": sum(p["det_shape"] == sh for p in ps),
                                 "confused_as": {k: sum(p["det_shape"] == k for p in ps)
                                                 for k in ("cube", "cylinder", "triangle", None) if k != sh}}
        out["per_shape"] = per_shape
        return out
    res = {"all": s(rows)}
    if by_family:
        for fam in sorted({r["family"] for r in rows}):
            for mode in sorted({r["mode"] for r in rows if r["family"] == fam}):
                rs = [r for r in rows if r["family"] == fam and r["mode"] == mode]
                res[f"{fam}-{mode}"] = s(rs)
    return res


def a2_report(rows) -> dict | None:
    """A2 slices only: the two objects of each close same-colour pair, and each occluded target split at 50 %
    hidden. Per group: n, found (matched within the 30 mm gate), xy p95 / max of found, xy > 10 mm (grasp would
    miss), colour unk / wrong, shape unk / wrong."""
    groups = {}
    for r in rows:
        by_gt = {p["gt"]: p for p in r["pairs"]}
        sel = []
        if r.get("close_pair"):
            sel = [("close pair", j) for j in r["close_pair"]]
        if r.get("occl_target") is not None:
            j = r["occl_target"]
            sel = [("occluded >50%" if r["hidden"][j] > 0.5 else "occluded <=50%", j)]
        for g, j in sel:
            groups.setdefault(g, []).append(by_gt.get(j))
    if not groups:
        return None
    out = {}
    for g, ps in groups.items():
        f = [p for p in ps if p is not None]
        xy = np.array([p["xy_mm"] for p in f]) if f else np.zeros(1)
        out[g] = {"n": len(ps), "found": len(f), "xy_p95": float(np.percentile(xy, 95)), "xy_max": float(xy.max()),
                  "xy_gt10": int((xy > 10).sum()) if f else 0,
                  "colour_unk": sum(p["det_color"] is None for p in f),
                  "colour_wrong": sum(p["det_color"] not in (None, p["gt_color"]) for p in f),
                  "shape_unk": sum(p["det_shape"] is None for p in f),
                  "shape_wrong": sum(p["det_shape"] not in (None, p["gt_shape"]) for p in f)}
    return out


def fmt(name, s):
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    return (f"{name:<16} n={s['frames']:>3} gt={s['gt']:>4} recall={100 * s['recall']:5.1f}% "
            f"fp={s['fp']:>3} ({100 * s['fp_rate']:4.1f}%) colour={100 * s['color_acc']:5.1f}% "
            f"(unk {s.get('color_unknown', 0)}, wrong {s.get('color_wrong', 0)}) "
            f"shape={100 * s['shape_acc']:5.1f}% (unk {s['shape_unknown']}, wrong {s['shape_wrong']}) "
            f"xy med/p95/max={f(s['xy_med_mm'])}/{f(s['xy_p95_mm'])}/{f(s['xy_max_mm'])} mm "
            f"yaw med/p95={f(s['yaw_med_deg'])}/{f(s['yaw_p95_deg'])} deg (missing {s['yaw_missing']}/{s['yaw_n']}) "
            f"ms cold med/p95={f(s['ms_med'])}/{f(s['ms_p95'])} warm={f(s.get('ms_warm_med'))}/"
            f"{f(s.get('ms_warm_p95'))} | hidden>50%: matched {s['hard_matched']}, "
            f"shape wrong {s['hard_shape_wrong']} | pts/obj type={s.get('score_type_per_obj', 0):+.3f} "
            f"colour={s.get('score_colour_per_obj', 0):+.3f}")


def draw_overlay(per, rgb, depth, gt_objects=None):
    """Debug image (BGR): RGB with detections (green: cross, label colour+shape, fitted outline at the top face),
    rejected blobs (magenta reason), GT outlines (red; objects with a yaw) or GT centres (red circle; no yaw), and
    next to it the depth validity map (white = valid). Returns (image, detections, rejected)."""
    import cv2
    from aaltonomous.baselines.classical.shape_fit import outline
    sc = per.scene
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    dets, rej = per.detect(rgb, depth, debug=True)
    for g in gt_objects or []:
        top = np.asarray(g["top"], float)
        if g.get("yaw") is not None:
            poly = outline(sc, g["shape"], g["yaw"]) + top[:2]
            uv = per.cal.project(per.cam, np.c_[poly, np.full(len(poly), top[2])])[:, :2]
            cv2.polylines(img, [np.round(uv).astype(np.int32)], True, (0, 0, 255), 1)
        else:
            u, v, _ = per.cal.project(per.cam, top)
            cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), 1)
    for d in dets:
        if not np.all(np.isfinite(d.top)):
            continue
        u, v, _ = per.cal.project(per.cam, d.top)
        cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        sh = getattr(d, "shape", None)
        if sh is not None:
            poly = outline(sc, sh, d.yaw or 0.0) + d.top[:2]
            uv = per.cal.project(per.cam, np.c_[poly, np.full(len(poly), d.top[2])])[:, :2]
            cv2.polylines(img, [np.round(uv).astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(img, f"{d.color[:1]}{(sh or '?')[:3]}", (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    for d, why in rej:
        if d is not None and np.all(np.isfinite(d.top)):
            u, v, _ = per.cal.project(per.cam, d.top)
            cv2.putText(img, why[:10], (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    valid = np.where(depth > 0, 200, 0).astype(np.uint8) if depth is not None else np.zeros(rgb.shape[:2], np.uint8)
    return np.hstack([img, cv2.cvtColor(valid, cv2.COLOR_GRAY2BGR)]), dets, rej


def overlay(variant, fid, out_dir: Path, extra=None):
    """draw_overlay for one frame id of a split; written to out_dir/<variant>_<id>.png."""
    import cv2
    per = make_perception(variant, extra)
    rgb, depth, gt = load_frame(fid)
    if VARIANTS[variant].get("_drop_depth"):
        depth = np.zeros_like(depth)
    img, _, _ = draw_overlay(per, rgb, depth, gt["objects"])
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{variant}_{Path(fid).name}.png"
    cv2.imwrite(str(p), img)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variant", nargs="+", default=["v2"], choices=sorted(VARIANTS))
    ap.add_argument("--set", default=None, help='extra config overrides as json, e.g. \'{"perception.top_band": 0.008}\'')
    ap.add_argument("--out", default=None, help="write summaries + per-frame rows (json)")
    ap.add_argument("--overlays", type=int, default=0, help="write debug overlays of the N worst frames")
    ap.add_argument("--failures", type=int, default=15, help="print up to N failure lines")
    ap.add_argument("--workers", type=int, default=8, help="render missing frames with this many processes")
    a = ap.parse_args(argv)
    ids = split_ids(a.split)
    render_split(a.split, a.workers)
    extra = json.loads(a.set) if a.set else None
    out = {"split": a.split, "n_frames": len(ids), "set": extra, "variants": {}}
    for v in a.variant:
        summ, rows = evaluate(v, ids, extra)
        out["variants"][v] = {"summary": summ, "rows": rows}
        print(f"\n=== {v} on {a.split} ({len(ids)} frames, SYNTHETIC unless ids start with rec:)")
        for k, s in summ.items():
            print(fmt(k, s))
        a2 = a2_report(rows)
        if a2:
            summ["a2"] = a2
            for g, q in a2.items():
                print(f"   A2 {g:<15} n={q['n']:>3} found={q['found']:>3} xy p95/max={q['xy_p95']:.1f}/{q['xy_max']:.1f} mm "
                      f"(>10 mm: {q['xy_gt10']}) colour unk/wrong={q['colour_unk']}/{q['colour_wrong']} "
                      f"shape unk/wrong={q['shape_unk']}/{q['shape_wrong']}")
        for sh, ps in summ["all"]["per_shape"].items():
            print(f"   {sh:<9} n={ps['n']:>3} ok={ps['shape_ok']:>3} confused={ps['confused_as']}")
        fails = []
        for r in rows:
            for j in r["miss"]:
                fails.append((r["id"], f"MISS gt#{j}"))
            for d in r["fp_desc"]:
                fails.append((r["id"], f"FP {d}"))
            for p in r["pairs"]:
                if not p["color_ok"]:
                    fails.append((r["id"], f"COLOUR {p['gt_color']}->{p['det_color']}"))
                if p["det_shape"] != p["gt_shape"]:
                    fails.append((r["id"], f"SHAPE {p['gt_shape']}->{p['det_shape']}"))
                if p["xy_mm"] > 15:
                    fails.append((r["id"], f"POS {p['xy_mm']:.1f} mm ({p['gt_shape']}, {p['method']})"))
                if p["yaw_err"] is not None and p["yaw_err"] > 10:
                    fails.append((r["id"], f"YAW {p['yaw_err']:.1f} deg ({p['gt_shape']})"))
        for fid, what in fails[: a.failures]:
            print(f"   {fid:<22} {what}")
        if len(fails) > a.failures:
            print(f"   ... {len(fails) - a.failures} more failure lines")
        if a.overlays:
            bad = sorted(rows, key=lambda r: -(len(r["miss"]) * 3 + len(r["fp"]) * 3 +
                                              sum(p["det_shape"] != p["gt_shape"] for p in r["pairs"])))
            for r in bad[: a.overlays]:
                print("   overlay ->", overlay(v, r["id"], HERE / "overlays", extra))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, default=float))
        print("results ->", a.out)


if __name__ == "__main__":
    main()
