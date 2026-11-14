"""Look at REAL (or any recorded) head frames: run perception, write overlays, report what it found.

    python -m aaltonomous.baselines.classical.loop.vision.look_frames data/drop1/venue_1113      # drop-1 session
    python -m aaltonomous.baselines.classical.loop.vision.look_frames logs/classical/2026-11-20  # run.py frame logs
    python -m aaltonomous.baselines.classical.loop.vision.look_frames <dir> --variant legacy v2 --per-step 3

Inputs it understands:
  drop-1 session (aaltonomous/data/drop1/capture.py): <session>/<step>/head_rgb/NNNNNN.png (RGB) +
      head_depth/NNNNNN.png (uint16 mm). Steps with known placements get APPROXIMATE ground truth from the
      session's checklist.yaml: `obj_*` steps (colour, shape, mat x / y; yaw unknown) and `layout_<seed>` steps
      (the printed layout card = scene/cards.make_card(seed); with yaw). Staff placement error (~5 mm, a few deg)
      is part of the measured error. Other steps (empty counter, lights) have no objects: every detection is an FP.
  *.npz frames: keys rgb/depth (run.py logs) or head_rgb/head_depth (capture_sim.py); a json sidecar with
      "objects" is used as ground truth when present.

Outputs: <out>/<variant>_<step>_<frame>.png overlays (see eval_vision.draw_overlay) and <out>/summary.json; a table
on stdout with detections, camera refinement (plane tilt / height, bin yaw / shift / rms), valid depth and, where
ground truth exists, matches, colour / shape correctness and xy error. These frames are a SANITY set: they are
looked at by eye every round, not tuned on.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .eval_vision import VARIANTS, draw_overlay, make_perception, match  # noqa: E402

HERE = Path(__file__).resolve().parent


def _drop1_gt(step_cfg, scene):
    """Approximate GT objects for a drop-1 checklist step, or [] (no objects) / None (unknown)."""
    if step_cfg is None:
        return None
    if step_cfg.get("kind") == "objects":
        out = []
        for o in step_cfg["objects"]:
            h = scene.object_height(o["shape"])
            out.append({"color": o["color"], "shape": o["shape"], "yaw": None, "visible_px": 9999,
                        "unoccluded_px": 9999, "top": list(scene.mat_to_pelvis([o["x"], o["y"], h]))})
        return out
    if step_cfg.get("kind") == "layout":
        from aaltonomous.scene.cards import make_card
        card = make_card(scene, step_cfg.get("task", "sort_fixed"), int(step_cfg["seed"]))
        out = []
        for c in card["cubes"]:
            h = scene.object_height(c.get("shape"))
            out.append({"color": c["color"], "shape": c.get("shape", "cube"), "yaw": -np.pi / 2 - c["yaw"],
                        "visible_px": 9999, "unoccluded_px": 9999,
                        "top": list(scene.mat_to_pelvis([c["x"], c["y"], h]))})
        return out
    return [] if step_cfg.get("kind") in ("empty", "scene", "lights") else None


def find_frames(root: Path, per_step: int):
    """[(name, rgb, depth, gt or None)] for a drop-1 session or a directory of .npz frames."""
    import cv2
    import yaml
    from aaltonomous.scene.scene import Scene
    out = []
    steps = sorted(p for p in root.iterdir() if (p / "head_rgb").is_dir()) if root.is_dir() else []
    if steps:
        ck = root / "checklist.yaml"
        if not ck.exists():
            ck = Path(__file__).resolve().parents[4] / "data" / "drop1" / "checklist.yaml"
        cfg = yaml.safe_load(ck.read_text())
        by_id = {s["id"]: s for s in cfg.get("steps", cfg if isinstance(cfg, list) else [])} \
            if isinstance(cfg, (dict, list)) else {}
        sc_path = root / "scene.yaml"
        scene = Scene.load(sc_path if sc_path.exists() else None)
        for st in steps:
            pngs = sorted((st / "head_rgb").glob("*.png"))
            if not pngs:
                continue
            pick = [pngs[int(round(k))] for k in np.linspace(0, len(pngs) - 1, min(per_step, len(pngs)))]
            gt = _drop1_gt(by_id.get(st.name), scene)
            for f in pick:
                rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                dpath = st / "head_depth" / f.name
                depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED) if dpath.exists() else None
                out.append((f"{st.name}_{f.stem}", rgb, depth, gt))
        return out
    for f in sorted(root.glob("*.npz") if root.is_dir() else [root]):
        z = np.load(f)
        rgb = z["rgb"] if "rgb" in z else z["head_rgb"]
        depth = z["depth"] if "depth" in z else z.get("head_depth")
        side = f.with_suffix(".json")
        gt = json.loads(side.read_text()).get("objects") if side.exists() else None
        out.append((f.stem, rgb, depth, gt))
    return out


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--variant", nargs="+", default=["v2"], choices=sorted(VARIANTS))
    ap.add_argument("--per-step", type=int, default=2, help="drop-1: frames per step (first ... last)")
    ap.add_argument("--out", default=str(HERE / "overlays" / "look"))
    a = ap.parse_args(argv)
    cv2.setNumThreads(1)
    frames = find_frames(Path(a.path), a.per_step)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for v in a.variant:
        per = make_perception(v)
        for name, rgb, depth, gt in frames:
            per.reset_calibration()             # frames may come from different sessions: no cached camera fix
            img, dets, rej = draw_overlay(per, rgb, depth, gt)
            cv2.imwrite(str(out / f"{v}_{name}.png"), img)
            pl = per.last_plane or {}
            b = pl.get("bins") or {}
            row = {"variant": v, "frame": name, "valid_depth": float((depth > 0).mean()) if depth is not None else 0.0,
                   "detections": [repr(d) for d in dets], "rejected": [w for _, w in rej],
                   "plane": {k: pl.get(k) for k in ("applied", "tilt_deg", "dz", "why")},
                   "bins": {k: b.get(k) for k in ("applied", "yaw_deg", "dxy", "rms_mm", "why")}}
            line = (f"{v:<7} {name:<28} dets={len(dets)} depth={row['valid_depth']:.2f} "
                    f"plane={'ok' if pl.get('applied') else pl.get('why')} tilt={pl.get('tilt_deg') or 0:.2f} "
                    f"dz={1000 * (pl.get('dz') or 0):+.1f}mm bins={'ok' if b.get('applied') else b.get('why')} "
                    f"yaw={b.get('yaw_deg') or 0:+.2f} rms={b.get('rms_mm') or 0:.1f}")
            if gt is not None:
                m = match(dets, gt)
                row["gt_n"], row["matched"] = len(gt), len(m)
                row["colour_ok"] = sum(dets[i].color == gt[j]["color"] for i, j, _ in m)
                row["shape_ok"] = sum(dets[i].shape == gt[j]["shape"] for i, j, _ in m)
                row["shape_wrong"] = sum(dets[i].shape not in (None, gt[j]["shape"]) for i, j, _ in m)
                row["xy_mm"] = [round(dd * 1000, 1) for _, _, dd in m]
                row["fp"] = len(dets) - len(m)
                line += (f" | GT {len(m)}/{len(gt)} colour {row['colour_ok']} shape {row['shape_ok']} "
                         f"(wrong {row['shape_wrong']}) FP {row['fp']} xy {row['xy_mm']}")
            print(line)
            summary.append(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"overlays + summary.json in {out}")


if __name__ == "__main__":
    main()
