"""Extra synthetic frame kinds for PERCEPTION_MILESTONES A2 / A3, built ON TOP of synth.py without changing it (so
every frozen synth.py frame and manifest stays bit-exact).

Frame ids keep the "<family>-<mode>-<seed>" form of dataset.py:

    <base>-close-<seed>   A2: two objects of the SAME colour placed 4.7-6.0 cm apart (centre to centre), not
                          overlapping. Base layout = synth.py "<base>-wide-<seed>" (base: train | unseen).
    <base>-occl-<seed>    A2: one object partly hidden by a dark occluder (a hand) hovering above it; the hidden
                          fraction is drawn uniformly in 0.3-0.9, so the slice has objects on both sides of the
                          "half hidden" line that eval_vision.detectable() uses.
    shift-<mode>-<seed>   A3: a harder appearance family than "unseen" (mode band | wide): strong coloured light
                          (deep tungsten / LED cyan / green / magenta casts), very dim or over-bright exposure, more
                          saturated mats incl. mats in the object hues, larger object colour shifts. Built from the
                          synth.py "unseen-<mode>-<seed>" spec, so geometry + sensor + distractors are unseen-like.

SYNTHETIC: algorithmic evidence only (see synth.py), never a stop-rule result.
"""
from __future__ import annotations

import colorsys
import copy

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from . import synth
from .synth import _jit_hsv, _u, in_view, outline, sample_spec

EXTRA_MODES = ("close", "occl")
EXTRA_FAMILIES = ("shift",)
_TAG = {"close": 101, "occl": 102, "shift": 103}


def is_extra(fid_family: str, fid_mode: str) -> bool:
    return fid_mode in EXTRA_MODES or fid_family in EXTRA_FAMILIES or fid_family.startswith("pose")


def _mat_poly(sc: Scene, ob: dict) -> np.ndarray:
    """Outline of a layout object in MAT coordinates (x, y), float32 (for cv2 overlap tests)."""
    # synth.object_planes: pelvis yaw = -pi/2 - layout yaw; outline() is in pelvis axes. Mat x = -pelvis y,
    # mat y = pelvis x (p_pelvis = (0.14 + y_mat, -0.45 + x_mat)): so mat (x, y) = (pelvis y, pelvis x) up to sign.
    p = outline(sc, ob.get("shape", "cube"), -np.pi / 2 - float(ob["yaw"]))      # pelvis-axis offsets (N,2)
    return np.stack([ob["x"] + p[:, 1], ob["y"] + p[:, 0]], 1).astype(np.float32)


def _overlap(sc, a: dict, b: dict, margin: float = 0.003) -> bool:
    import cv2
    pa, pb = _mat_poly(sc, a), _mat_poly(sc, b)
    # grow b slightly around its centre for a safety margin
    c = pb.mean(0)
    pb = c + (pb - c) * (1 + margin / max(np.linalg.norm(pb - c, axis=1).max(), 1e-6))
    area, _ = cv2.intersectConvexConvex(pa, pb.astype(np.float32))
    return area > 0


def _close(spec: dict, rng, sc: Scene, cal: Calibration) -> dict:
    """Move one object next to another and give it the same colour (<= 2 per colour kept)."""
    obs = spec["objects"]
    for _ in range(400):
        i, j = (int(k) for k in rng.choice(len(obs), 2, replace=False))
        d, a = _u(rng, 0.047, 0.060), _u(rng, 0, 2 * np.pi)
        cand = {**obs[j], "x": obs[i]["x"] + d * np.cos(a), "y": obs[i]["y"] + d * np.sin(a)}
        if not in_view(sc, cal, cand["x"], cand["y"], 14.0):
            continue
        if any(_overlap(sc, cand, o) for k, o in enumerate(obs) if k != j):
            continue
        if any(np.hypot(cand["x"] - o["x"], cand["y"] - o["y"]) < 0.065 for k, o in enumerate(obs) if k not in (i, j)):
            continue
        col, old = obs[i]["color"], obs[j]["color"]
        new = copy.deepcopy(obs)
        new[j] = cand
        new[j]["color"] = col
        for k, o in enumerate(new):         # keep at most 2 of one colour: a third of `col` takes j's old colour
            if k not in (i, j) and o["color"] == col:
                o["color"] = old
                o["albedo"] = _jit_hsv(sc.colors[old], rng, 0.02, (.8, 1.15), (.8, 1.15))
        new[j]["albedo"] = _jit_hsv(sc.colors[col], rng, 0.02, (.8, 1.15), (.8, 1.15))
        spec = {**spec, "objects": new, "close_pair": [i, j], "close_dist": float(d)}
        spec["layout"] = [{k: v for k, v in o.items() if k != "albedo"} for o in new]
        return spec
    raise RuntimeError("no close pair found")


def _hidden_frac(sc, cal, ob: dict, oc: dict) -> float:
    """Fraction of the object's image silhouette covered by the occluder's silhouette (nominal camera, convex
    hulls of projected corners; the occluder is above the object, so overlap = hidden)."""
    import cv2
    h = sc.object_height(ob.get("shape"))
    poly = _mat_poly(sc, ob).astype(float)
    pts = np.array([[x, y, z] for x, y in poly for z in (0.0, h)])
    half = np.asarray(oc["size"], float) / 2
    box = np.array([[oc["x"] + sx * half[0], oc["y"] + sy * half[1], oc["z0"] + sz * 2 * half[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    def mask(p):
        uv = cal.project("head", sc.mat_to_pelvis(p))[:, :2]
        m = np.zeros((synth.H_IMG, synth.W_IMG), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        return m
    mo, mc = mask(pts), mask(box)
    return float((mo & mc).sum() / max(mo.sum(), 1))


def _occl(spec: dict, rng, sc: Scene, cal: Calibration) -> dict:
    obs = spec["objects"]
    target = _u(rng, 0.3, 0.9)
    best = None
    for _ in range(600):
        t = int(rng.integers(len(obs)))
        ob = obs[t]
        oc = {"x": ob["x"] + _u(rng, -.05, .05), "y": ob["y"] + _u(rng, -.08, .0), "z0": _u(rng, .06, .11),
              "size": [_u(rng, .04, .09), _u(rng, .05, .10), .03], "rgb": [_u(rng, .08, .25)] * 3}
        f = _hidden_frac(sc, cal, ob, oc)
        if best is None or abs(f - target) < abs(best[0] - target):
            best = (f, t, oc)
        if abs(f - target) < 0.03:
            break
    f, t, oc = best
    return {**spec, "occluders": list(spec["occluders"]) + [oc], "occl_target": t, "occl_frac_est": f}


def _shift(spec: dict, rng, sc: Scene) -> dict:
    """Push the appearance of an `unseen` spec further than any synth.py family."""
    spec = copy.deepcopy(spec)
    style = str(rng.choice(["deep_warm", "cyan", "green", "magenta", "very_dim", "over_bright"]))
    tint = {"deep_warm": [1.0, .68, .42], "cyan": [.55, .85, 1.0], "green": [.7, 1.0, .65],
            "magenta": [1.0, .62, .9]}.get(style, [1.0, 1.0, 1.0])
    tint = [c * (1 + _u(rng, -.05, .05)) for c in tint]
    L = spec["light"]
    L.update(style=f"shift_{style}", key_rgb=tint, ambient_rgb=[c * (1 + _u(rng, -.05, .05)) for c in tint])
    if style == "very_dim":
        L["key"], L["ambient"] = 0.62 * _u(rng, .18, .3), 0.36 * _u(rng, .18, .3)
    elif style == "over_bright":
        L["key"], L["ambient"] = 0.62 * _u(rng, 2.0, 2.6), 0.36 * _u(rng, 1.8, 2.4)
    tex = spec["texture"]
    k = str(rng.choice(["solid_hue", "checker_hue", tex["kind"]]))
    if k in ("solid_hue", "checker_hue"):      # mats in (or near) the object hues: colour alone cannot segment
        hues = [0.0, 0.14, 0.62]
        tex.update(kind="solid" if k == "solid_hue" else "checker",
                   rgb_a=list(colorsys.hsv_to_rgb(hues[int(rng.integers(3))] + _u(rng, -.03, .03), _u(rng, .45, .8),
                                                 _u(rng, .45, .9))),
                   rgb_b=list(colorsys.hsv_to_rgb(hues[int(rng.integers(3))] + _u(rng, -.03, .03), _u(rng, .3, .8),
                                                 _u(rng, .35, .9))))
    for ob in spec["objects"]:
        ob["albedo"] = _jit_hsv(sc.colors[ob["color"]], rng, 0.06, (.5, 1.25), (.45, 1.3))
    S = spec["sensor"]
    S["rgb_gain"] = _u(rng, .6, 1.4)
    spec.update(family="shift", shift_style=style)
    return spec


def _pose(spec: dict, rng, k: float) -> dict:
    """A4: camera pose error of up to +-k cm (each axis) and +-k deg (each axis), uniform; train otherwise."""
    spec = copy.deepcopy(spec)
    spec["camera"] = {"dt": [_u(rng, -k / 100, k / 100) for _ in range(3)], "drot_deg": [_u(rng, -k, k) for _ in range(3)]}
    spec["family"] = f"pose{k:g}"
    return spec


def sample_spec_extra(seed: int, family: str, mode: str, sc: Scene | None = None) -> dict:
    sc = sc or Scene.load()
    cal = Calibration.load()
    if family.startswith("pose"):      # "pose<k>": k in cm / deg (ids like pose3-band-123; "p" for a decimal point)
        k = float(family[4:].replace("p", "."))
        rng = np.random.default_rng([int(seed), 104, int(round(k * 10))])
        return _pose(sample_spec(seed, "train", mode, sc), rng, k)
    if family in EXTRA_FAMILIES:
        if mode not in synth.LAYOUT_MODES:
            raise ValueError(f"shift family needs mode band | wide, not {mode!r}")
        rng = np.random.default_rng([int(seed), _TAG[family], synth.LAYOUT_MODES.index(mode)])
        return _shift(sample_spec(seed, "unseen", mode, sc), rng, sc)
    if mode not in EXTRA_MODES:
        raise ValueError(mode)
    rng = np.random.default_rng([int(seed), _TAG[mode], synth.FAMILIES.index(family)])
    base = sample_spec(seed, family, "wide", sc)
    if len(base["objects"]) < 2:
        raise RuntimeError("layout with < 2 objects")
    spec = _close(base, rng, sc, cal) if mode == "close" else _occl(base, rng, sc, cal)
    spec["layout_mode"] = mode
    return spec


def check():
    """Self-check (seconds): close pairs are same-colour, 4.7-6 cm apart, non-overlapping, <= 2 per colour;
    occluded targets hit their drawn hidden fraction in the GT render within 0.15; shift specs differ from unseen."""
    sc = Scene.load()
    for s in range(6):
        sp = sample_spec_extra(s, "train", "close", sc)
        i, j = sp["close_pair"]
        a, b = sp["objects"][i], sp["objects"][j]
        assert a["color"] == b["color"] and 0.047 <= np.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= 0.0601
        assert not _overlap(sc, a, b, 0.0)
        cols = [o["color"] for o in sp["objects"]]
        assert max(cols.count(c) for c in tasks.COLORS) <= 2, cols
    r = synth.SynthRenderer(sc)
    for s in range(3):
        sp = sample_spec_extra(s, "train", "occl", sc)
        _, _, gt = r.render(sp)
        g = gt["objects"][sp["occl_target"]]
        frac = 1 - g["visible_px"] / max(g["unoccluded_px"], 1)
        assert abs(frac - sp["occl_frac_est"]) < 0.2, (s, frac, sp["occl_frac_est"])
    a, b = sample_spec(3, "unseen", "band", sc), sample_spec_extra(3, "shift", "band", sc)
    assert [o["x"] for o in a["objects"]] == [o["x"] for o in b["objects"]] and a["light"] != b["light"]
    print("synth_extra check OK")


if __name__ == "__main__":
    check()
