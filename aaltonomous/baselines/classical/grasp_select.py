"""Grasp selection from the object's dimensions and the hand's geometry (no per-object coordinates).

Dex3 facts used (from the URDF through ArmKinematics.tips_in_palm, closed posture): index and middle close on
PARALLEL lines `spacing` apart (57 mm, no abduction joint), the thumb closes on the line midway between them.
Across those two lines an object of width w can sit in three ways:

    w >= spacing + pad         both finger pads land fully on the object  -> TRIPOD: object centred (lateral 0),
                               thumb opposite, three contacts, no couple.
    spacing - pad < w < spacing + pad
                               centred, the object touches only the INNER EDGES of the two fingers: it is pushed
                               into the gap and wedges there (seen in Isaac: carried home in the open hand; and
                               measured: centred grasps V3_tri3face 4/20 vs 17/20, V4_cylcage 9/20 vs 19/20 on
                               grasp-variants). Never centre such an object.
    w <= spacing - pad         centred, it passes between the fingers untouched.

In the last two cases the grasp is a PINCH of the thumb against ONE finger: the object is shifted sideways
(`lateral`, palm z) so that this finger's pad is fully on a flat face and the other finger passes beside the object
with `clearance`. Height: the middle of the object's own height (scene.object_dims), raised by the planner only if
the fingertips would touch the counter.

The scene's objects are all 45 mm wide: pinch, lateral inside [8.5, 28.5] mm -> the configured 15 mm is kept.
A 90 mm object would get the centred three-finger grasp.
"""
from __future__ import annotations

import numpy as np


def hand_geometry(kin, postures, pad_width):
    """Finger lines of the closed LEFT hand in the palm frame (the right hand mirrors y only)."""
    t = kin.tips_in_palm("left", postures["closed"])      # rows: index tip, middle tip, thumb tip
    zi, zm, zt = float(t[0][2]), float(t[1][2]), float(t[2][2])
    return {"spacing": abs(zi - zm), "centre": 0.5 * (zi + zm), "pinch_sign": 1.0 if zi >= zm else -1.0,
            "thumb": zt, "pad": float(pad_width)}


def across_width(shape, dims):
    """(min, max) extent of the object across the finger lines over all yaws (m). Unknown shape: scene width."""
    w = float(dims["width"])
    if shape == "cube":
        return w, w * np.sqrt(2.0)
    if shape == "triangle":
        return w, float(dims.get("side", w))
    return w, w


def select(shape, dims, hand, cfg):
    """-> {"type": "tripod" | "pinch", "lateral": palm-z offset of the object centre (m, towards the pinch finger),
    "height_frac": 0.5, "wedge_if_centred": bool, "why": str}. cfg = config `grasp.select`."""
    pad, s = hand["pad"], hand["spacing"]
    clear, fillet = float(cfg.get("clearance", 0.003)), float(dims.get("fillet", 0.0))
    w_min, w_max = across_width(shape, dims)
    out = {"height_frac": float(cfg.get("height_frac", 0.5)), "width": w_min,
           "wedge_if_centred": bool(s - pad < w_min < s + pad)}
    if w_min >= s + pad:
        return {**out, "type": "tripod", "lateral": 0.0,
                "why": f"width {1e3 * w_min:.0f} mm >= finger spacing {1e3 * s:.0f} + pad {1e3 * pad:.0f} mm: both pads on the object"}
    # pinch: the pinch finger's centre line must be on the flat face, the other finger must pass beside the object
    lo_face = s / 2 - (w_min / 2 - fillet)                 # pinch finger line no further out than the face edge
    lo_other = w_min / 2 + clear + pad / 2 - s / 2         # other finger's inner edge clears the object
    lo, hi = max(lo_face, lo_other, 0.0), s / 2            # hi: object centre not beyond the pinch finger line
    want = float(cfg.get("pinch_lateral", 0.015))
    lat = float(np.clip(want, lo, hi))
    kind = "wedges between the fingers" if out["wedge_if_centred"] else "passes between the fingers"
    return {**out, "type": "pinch", "lateral": lat,
            "why": f"width {1e3 * w_min:.0f} mm, centred it {kind} (spacing {1e3 * s:.0f}, pad {1e3 * pad:.0f} mm): "
                   f"thumb vs one finger, object {1e3 * lat:.1f} mm off the hand centre line "
                   f"(allowed {1e3 * lo:.1f}-{1e3 * hi:.1f})"}
