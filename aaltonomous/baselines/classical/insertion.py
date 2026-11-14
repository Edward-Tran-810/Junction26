"""Shape-matching insertion geometry (rung 3 `insert_shapes`): which hole, where, how the object must be turned,
how much pose error each shape tolerates. Everything is derived from scene.yaml / scene.Scene (hole polygons come from
Scene.lid_holes(), the same function the simulator's lid mesh and the scorer use); nothing is hardcoded.

Frames: pelvis frame throughout (+x forward, +y left, z up). The mat -> pelvis map swaps x / y (a mirror), so each
hole's yaw is found NUMERICALLY in the convention perception uses for object yaw (the canonical outline
`scene.shape_polygon(shape)`, apex +y, rotated by yaw about pelvis +z), not assumed.

Shape -> hole is by shape only (colour is irrelevant here): cube -> square, cylinder -> circle, triangle -> triangle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aaltonomous.scene.scene import in_convex, shape_polygon

SHAPE_TO_HOLE = {"cube": "square", "cylinder": "circle", "triangle": "triangle"}   # object shape -> hole shape name
SYMMETRY = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3, "cylinder": None}        # yaw period of the object outline


def wrap(a, period):
    """a wrapped to (-period/2, period/2]."""
    return (a + period / 2) % period - period / 2


@dataclass
class Hole:
    shape: str            # object shape that goes through it
    name: str             # square | circle | triangle
    centre: np.ndarray    # (2,) pelvis xy
    poly: np.ndarray      # (N, 2) pelvis xy
    yaw: float            # perception-convention yaw of an object that fits exactly (wrapped to its symmetry)
    apothem: float        # of the hole outline (object apothem + clearance / 2)


class InsertGeometry:
    def __init__(self, scene):
        self.scene = scene
        self.lid_under_z = float(scene.mat_to_pelvis(np.r_[0.0, 0.0, scene.lid_z[0]])[2])
        self.lid_top_z = float(scene.mat_to_pelvis(np.r_[0.0, 0.0, scene.lid_z[1]])[2])
        self.slack = float(scene.cfg["lid"]["hole_clearance"]) / 2
        self.holes = {}
        for h in scene.lid_holes():
            c = scene.mat_to_pelvis(np.r_[h["center"], 0.0])[:2]
            P = scene.mat_to_pelvis(np.c_[h["poly"], np.zeros(len(h["poly"]))])[:, :2]
            if _area(P) < 0:                       # the mat -> pelvis mirror flips the winding; in_convex wants CCW
                P = P[::-1]
            a = scene.object_dims(h["shape"])["apothem"] + self.slack
            self.holes[h["shape"]] = Hole(h["shape"], SHAPE_TO_HOLE[h["shape"]], c, P, self._find_yaw(h["shape"], c, P, a), a)

    @staticmethod
    def _find_yaw(shape, c, P, a):
        per = SYMMETRY[shape]
        if per is None:
            return 0.0
        best = min(np.linspace(-per / 2, per / 2, 3601), key=lambda y: _poly_gap(c + shape_polygon(shape, a, y), P))
        return float(best)

    # ------------------------------------------------------------------ queries
    def hole_for(self, shape):
        """The matching hole, or None for an unknown shape (never a guess: the caller re-looks / skips)."""
        return self.holes.get(shape)

    def outline(self, shape, xy, yaw):
        return np.asarray(xy, float) + shape_polygon(shape, self.scene.object_dims(shape)["apothem"], yaw)

    def fits(self, shape, xy, yaw, hole_shape=None, margin=0.0):
        """True if the object's whole outline (at xy, yaw) lies inside the hole polygon shrunk by `margin` (m)."""
        h = self.holes[hole_shape or shape]
        poly = h.poly
        if margin:
            c = poly.mean(0)
            d = poly - c
            poly = c + d * (1 - margin / np.min(np.linalg.norm(d, axis=1) * np.cos(np.pi / len(poly))))
        return all(in_convex(poly, p) for p in self.outline(shape, xy, yaw))

    def align_delta(self, shape, obj_yaw, hole_shape=None):
        """Rotation (rad, about +z) to apply to an object of yaw `obj_yaw` so that it matches the hole; smallest
        |delta| over the object's symmetry. Cylinders: 0."""
        per = SYMMETRY[shape]
        if per is None:
            return 0.0
        return float(wrap(self.holes[hole_shape or shape].yaw - obj_yaw, per))

    def tolerance(self, shape, step=1e-4):
        """Largest error that still fits (outline fully inside the hole): xy radius at yaw 0 (worst direction),
        and yaw at xy 0. Pure geometry, perfect fillet-less outlines."""
        h = self.holes[shape]
        yaw0 = h.yaw
        r = 0.0
        while all(self.fits(shape, h.centre + r * np.array([np.cos(t), np.sin(t)]), yaw0) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)):
            r += step
        out = {"xy_m": round(r - step, 4)}
        if SYMMETRY[shape] is None:
            out["yaw_rad"] = None
        else:
            d = 0.0
            while self.fits(shape, h.centre, yaw0 + d) and self.fits(shape, h.centre, yaw0 - d) and d < SYMMETRY[shape]:
                d += np.radians(0.25)
            out["yaw_rad"] = round(float(d - np.radians(0.25)), 4)
        return out


def _poly_gap(A, B):
    """Sum over vertices of A of the distance to the nearest vertex of B (equal polygons -> 0)."""
    return float(sum(np.min(np.linalg.norm(B - a, axis=1)) for a in A))


def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
