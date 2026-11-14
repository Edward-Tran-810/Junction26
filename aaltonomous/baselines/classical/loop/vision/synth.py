"""Deterministic synthetic head RGB-D frames with exact ground truth, for OFFLINE perception development and tests.

    from aaltonomous.baselines.classical.loop.vision.synth import SynthRenderer, sample_spec
    r = SynthRenderer()                                   # scene.yaml + calibration.yaml (nominal values)
    spec = sample_spec(seed=7, family="train", layout_mode="band")
    rgb, depth, gt = r.render(spec)                       # (480,640,3) uint8, (480,640) uint16 mm, dict

    python -m aaltonomous.baselines.classical.loop.vision.synth --seed 7 --family unseen --out /tmp/x.png

What it is
    A small ray caster written for this test bench (numpy + OpenCV only). Every object is an exact convex prism of
    the competition geometry (scene.yaml: cube / cylinder / equilateral triangle prism, size 45 mm, height = size);
    the bins are exact boxes; the counter carries the real printed mat texture (scene/assets/mat_texture.png) or a
    procedural replacement. Shading = ambient + one point "key" light (Lambert) + a broad white sheen (matte PLA
    tops look washed out under an overhead light, as in the Isaac renders) + cast shadows of the objects on the
    counter. The sensor model adds RGB gain / noise / blur, depth noise (absolute + relative to range), random
    depth holes, invalid depth next to depth edges (stereo occlusion band of a D435) and RGB-depth misregistration.
    The camera can be rendered from a slightly WRONG pose while perception keeps the nominal calibration: that is
    how calibration error is simulated.

What it is NOT
    Not Isaac, not a real camera. Results on these frames are ALGORITHMIC evidence only: they show that the
    perception code does what it is meant to do under controlled disturbances. They are never evidence for the
    vision stop rule (which needs sim or real frames, see loop/STATE.md).

Ground truth conventions (the same as the Isaac scene service, sim/aalto_task/layout.py)
    centre   object centre in the pelvis frame (m); the object rests on the counter, so centre z = counter + H/2.
    top      top-face centre = centre + (0, 0, H/2). This is what Detection.top estimates.
    yaw      rotation about pelvis +z (rad) of the canonical outline scene.shape_polygon(shape, ...)
             (square with axis-aligned sides; triangle with its apex towards local +y). A layout yaw (mat frame)
             becomes pelvis yaw = -pi/2 - yaw (layout.mat_yaw_to_pelvis). Symmetry: cube 90 deg, triangle 120 deg,
             cylinder: none defined.

Families (all knobs are drawn from the seed, so (seed, family, layout_mode) fully defines a frame):
    nominal  sim-like lighting, real mat, light sensor noise, perfect calibration.
    train    ranges of sim/dr/train.yaml (lighting, mat tint, object HSV, sensor noise, +-1 cm / +-1 deg camera).
    unseen   things NOT in `train`, for held-out tests: procedural mats (wood, solid colour, checkerboard, blotches),
             flat coloured "stickers" in the object colours lying on the mat, a dark occluder (a hand) above the
             counter, coloured / dim / harsh low-angle light, stronger object colour shifts, RGB blur, depth
             misregistration, more depth holes.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
from pathlib import Path

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import SHAPES, Scene, shape_polygon

ASSETS = Path(__file__).resolve().parents[4] / "scene" / "assets"
H_IMG, W_IMG = 480, 640
FLOOR_Z = -0.79                     # floor in the pelvis frame (pelvis 0.79 m above the floor, scene.yaml)
CYLINDER_SIDES = 48                 # polygon used for the cylinder (max radial error 0.05 mm at 22.5 mm radius)
FAMILIES = ("nominal", "train", "unseen")
LAYOUT_MODES = ("band", "wide")

# material kinds (index into the per-frame material table is stored per pixel)
KIND_NONE, KIND_FLOOR, KIND_COUNTER, KIND_SOLID = 0, 1, 2, 3


# ----------------------------------------------------------------------------------------------- small geometry helpers
def rot_z(a: float) -> np.ndarray:
    """3x3 rotation about +z by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle(v: np.ndarray) -> np.ndarray:
    """Rotation matrix for the rotation vector v (axis * angle, rad); Rodrigues' formula."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def prism_planes(poly_xy: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Half-space form of a vertical prism over a CCW convex polygon (pelvis xy) between heights z0 < z1.

    Returns (K, 4) rows (n_x, n_y, n_z, d): a point p is inside iff n . p <= d for every row. Side rows use the
    outward edge normals of the CCW polygon (edge e = (ex, ey) -> outward normal (ey, -ex)); the last two rows are
    the top (z <= z1) and the bottom (-z <= -z0)."""
    p = np.asarray(poly_xy, float)
    e = np.roll(p, -1, axis=0) - p
    n = np.stack([e[:, 1], -e[:, 0], np.zeros(len(p))], -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = np.einsum("ij,ij->i", n[:, :2], p)
    side = np.c_[n, d]
    return np.vstack([side, [0, 0, 1.0, z1], [0, 0, -1.0, -z0]])


def box_planes(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Axis-aligned box (pelvis frame) as half-spaces, same row format as prism_planes."""
    c, h = np.asarray(center, float), np.asarray(half, float)
    rows = []
    for ax in range(3):
        n = np.zeros(3)
        n[ax] = 1.0
        rows.append(np.r_[n, c[ax] + h[ax]])
        rows.append(np.r_[-n, -(c[ax] - h[ax])])
    return np.array(rows)


def intersect_convex(o: np.ndarray, rays: np.ndarray, planes: np.ndarray):
    """Ray / convex-polyhedron intersection (Cyrus-Beck clipping), vectorised over rays.

    o (3,) ray origin, rays (M, 3) directions (any length; t is in units of that length), planes (K, 4).
    Returns (t_enter (M,), plane index of entry (M,)); t_enter = inf where the ray misses or starts inside.
    For every plane, n.(o + t r) <= d is a bound on t: an upper bound where n.r > 0 (leaving), a lower bound where
    n.r < 0 (entering). The ray hits iff the largest lower bound is <= the smallest upper bound and > 0."""
    n, d = planes[:, :3], planes[:, 3]
    denom = rays @ n.T                                   # (M, K)
    num = d - n @ o                                      # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / denom
    t_in = np.where(denom < 0, t, -np.inf)
    t_out = np.where(denom > 0, t, np.inf)
    parallel_out = (denom == 0) & (num < 0)              # parallel to a face and outside it: never inside
    k_in = t_in.argmax(1)
    t_enter = t_in.max(1)
    t_exit = t_out.min(1)
    hit = (t_enter <= t_exit) & (t_enter > 1e-9) & ~parallel_out.any(1)
    return np.where(hit, t_enter, np.inf), k_in


# ----------------------------------------------------------------------------------------------- frame specification
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _jit_hsv(rgb255, rng, hue_jit, sat_rng, val_rng):
    """Object albedo after the DR colour jitter (hue shift in turns, multiplicative saturation and value)."""
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(rgb255, float) / 255.0))
    h = (h + _u(rng, -hue_jit, hue_jit)) % 1.0
    s = min(1.0, s * _u(rng, *sat_rng))
    v = min(1.0, v * _u(rng, *val_rng))
    return list(colorsys.hsv_to_rgb(h, s, v))


def in_view(sc: Scene, cal: Calibration, x_mat: float, y_mat: float, margin_px: float = 6.0, cam: str = "head"):
    """True if an object (any shape and yaw: its bounding cylinder) at mat (x, y) projects fully inside the image
    of the NOMINAL camera, with `margin_px` to spare (the +-1 deg camera error moves the image by ~10 px)."""
    r = sc.object_dims("cube")["circumradius"]
    h = sc.object_height()
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = np.array([[x_mat + r * np.cos(a), y_mat + r * np.sin(a), z] for a in ang for z in (0.0, h)])
    uvz = cal.project(cam, sc.mat_to_pelvis(pts))
    c = cal.cams[cam]
    return bool(np.all(uvz[:, 2] > 0) and np.all(uvz[:, 0] >= margin_px) and np.all(uvz[:, 0] < c.width - margin_px)
                and np.all(uvz[:, 1] >= margin_px) and np.all(uvz[:, 1] < c.height - margin_px))


def _wide_layout(rng, sc: Scene, n_min=3, n_max=6, spacing=0.065):
    """Objects anywhere on the mat in front of the bins that the head camera sees completely (not only the layout
    band), 3-6 objects, 1-2 per colour like sample_layout. Mat coords; yaw in the layout (mat) convention."""
    cal = Calibration.load()
    n = int(rng.integers(n_min, n_max + 1))
    colors = list(tasks.COLORS) + list(rng.permutation(tasks.COLORS))[: max(0, n - 3)]
    rng.shuffle(colors)
    bins_near_y = min(c[1] for c in sc.cfg["bins"]["centers"].values()) - sc.cfg["bins"]["outer_size"][1] / 2
    x0, x1 = 0.12, sc.cfg["counter"]["mat_size"][0] - 0.12
    y0, y1 = 0.125, bins_near_y - 0.045           # near limit: whole object inside the head image
    out = []
    for _ in range(4000):
        if len(out) == len(colors):
            break
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if all(np.hypot(x - c["x"], y - c["y"]) >= spacing for c in out) and in_view(sc, cal, x, y, 14.0):
            out.append({"color": str(colors[len(out)]), "x": round(float(x), 3), "y": round(float(y), 3),
                        "yaw": round(float(rng.uniform(-np.pi / 4, np.pi / 4)), 3),
                        "shape": str(SHAPES[int(rng.integers(len(SHAPES)))])})
    return out


def sample_spec(seed: int, family: str = "train", layout_mode: str = "band", sc: Scene | None = None) -> dict:
    """Draw every random choice of one synthetic frame from `seed` (numpy default_rng). Pure function.

    Returns a JSON-serialisable dict: layout (mat coords, tasks.sample_layout format), lighting, mat texture,
    object albedos, distractors, sensor noise and camera pose error. `render()` turns it into images."""
    if family not in FAMILIES or layout_mode not in LAYOUT_MODES:
        raise ValueError(f"family {family!r} / layout_mode {layout_mode!r}")
    sc = sc or Scene.load()
    rng = np.random.default_rng([int(seed), FAMILIES.index(family), LAYOUT_MODES.index(layout_mode)])
    if layout_mode == "band":
        for attempt in range(5):         # sample_layout occasionally fails to meet the spacing: draw again
            try:
                layout = tasks.sample_layout(rng, **sc.layout_kwargs())
                break
            except RuntimeError:
                if attempt == 4:
                    raise
    else:
        layout = _wide_layout(rng, sc)
    nominal, unseen = family == "nominal", family == "unseen"

    # lighting (pelvis frame). Isaac: key light at world (0.5, 0, 2.2) = pelvis (0.5, 0, 1.41), dome ambient.
    if nominal:
        light = {"key_pos": [0.5, 0.0, 1.41], "key_rgb": [1.0, 0.97, 0.92], "key": 0.62, "ambient": 0.36,
                 "ambient_rgb": [1.0, 1.0, 1.0], "sheen": 0.5}
    elif not unseen:
        light = {"key_pos": [0.5 + _u(rng, -.4, .4), _u(rng, -.4, .4), 1.41 + _u(rng, -.3, .3)],
                 "key_rgb": [0.97 * (1 + _u(rng, -.08, .08)) for _ in range(3)], "key": 0.62 * _u(rng, .6, 1.4),
                 "ambient": 0.36 * _u(rng, .5, 1.5), "ambient_rgb": [1 + _u(rng, -.05, .05) for _ in range(3)],
                 "sheen": _u(rng, .25, .75)}
    else:
        style = rng.choice(["dim", "harsh_low", "tinted", "bright"])
        key_pos = [0.5 + _u(rng, -.6, .6), _u(rng, -.8, .8), 1.41 + _u(rng, -.4, .4)]
        key, amb = 0.62 * _u(rng, .5, 1.6), 0.36 * _u(rng, .4, 1.6)
        tint = [1.0, 1.0, 1.0]
        if style == "dim":
            key, amb = 0.62 * _u(rng, .3, .5), 0.36 * _u(rng, .3, .5)
        elif style == "harsh_low":                       # low-angle light: long shadows, dark top faces
            ang = _u(rng, 0, 2 * np.pi)
            key_pos = [0.3 + 0.9 * np.cos(ang), 0.9 * np.sin(ang), 0.25 + _u(rng, 0, .25)]
            key, amb = 0.62 * _u(rng, 1.0, 1.8), 0.36 * _u(rng, .2, .5)
        elif style == "tinted":                          # warm (tungsten) or cool (daylight LED) cast
            warm = rng.random() < 0.5
            tint = [1.0, 0.82, 0.62] if warm else [0.75, 0.88, 1.0]
            tint = [t * (1 + _u(rng, -.05, .05)) for t in tint]
        else:
            key, amb = 0.62 * _u(rng, 1.5, 2.0), 0.36 * _u(rng, 1.2, 1.8)
        light = {"style": str(style), "key_pos": key_pos, "key_rgb": tint, "key": key, "ambient": amb,
                 "ambient_rgb": [t * (1 + _u(rng, -.05, .05)) for t in tint], "sheen": _u(rng, .1, .9)}

    # counter / mat texture
    if unseen:
        kind = str(rng.choice(["wood", "solid", "checker", "blotch", "real_tinted"]))
        tex = {"kind": kind, "seed": int(rng.integers(1 << 31)),
               "rgb_a": list(colorsys.hsv_to_rgb(_u(rng, 0, 1), _u(rng, 0, .55), _u(rng, .35, .95))),
               "rgb_b": list(colorsys.hsv_to_rgb(_u(rng, 0, 1), _u(rng, 0, .55), _u(rng, .35, .95))),
               "period": _u(rng, .02, .10)}
        if kind == "wood":                              # brown / orange: the hard case for red and yellow masks
            tex["rgb_a"] = list(colorsys.hsv_to_rgb(_u(rng, .04, .11), _u(rng, .35, .65), _u(rng, .45, .85)))
            tex["rgb_b"] = [c * _u(rng, .6, .85) for c in tex["rgb_a"]]
    else:
        tex = {"kind": "real", "scale": 1.0 if nominal else _u(rng, .8, 1.15),
               "rgb_jit": [1.0] * 3 if nominal else [1 + _u(rng, -.06, .06) for _ in range(3)],
               "bias": 0.0 if nominal else _u(rng, -.03, .03)}
    tex["counter_rgb"] = [0.35 * (1 + (0 if nominal else _u(rng, -.2, .2))) for _ in range(3)]
    tex["bin_rgb"] = [c / 255 * (1 + (0 if nominal else _u(rng, -.08, .08))) for c in sc.bin_rgb]

    # objects: albedo (DR colour jitter) and placement error on top of the layout card
    hue_j, sat_r, val_r = (0.0, (1, 1), (1, 1)) if nominal else \
        ((0.02, (.8, 1.15), (.8, 1.15)) if not unseen else (0.035, (.65, 1.2), (.6, 1.25)))
    objects = []
    for ob in layout:
        r = _u(rng, 0, .005) if not nominal else 0.0
        a = _u(rng, 0, 2 * np.pi)
        objects.append({**ob, "x": ob["x"] + r * np.cos(a), "y": ob["y"] + r * np.sin(a),
                        "yaw": ob["yaw"] + (0.0 if nominal else np.radians(_u(rng, -5, 5))),
                        "albedo": _jit_hsv(sc.colors[ob["color"]], rng, hue_j, sat_r, val_r)})

    # distractors (unseen only): flat stickers in object colours lying on the mat, one dark occluder (a hand)
    stickers, occluders = [], []
    if unseen:
        for _ in range(int(rng.integers(0, 4))):
            c = str(rng.choice(tasks.COLORS))
            stickers.append({"color": c, "shape": str(rng.choice(SHAPES)), "x": _u(rng, .1, .8), "y": _u(rng, .03, .2),
                             "yaw": _u(rng, 0, 2 * np.pi), "size": _u(rng, .03, .06),
                             "albedo": _jit_hsv(sc.colors[c], rng, .02, (.8, 1.1), (.8, 1.1))})
        if rng.random() < 0.3 and objects:
            t = objects[int(rng.integers(len(objects)))]
            occluders.append({"x": t["x"] + _u(rng, -.04, .04), "y": t["y"] + _u(rng, -.06, -.01),
                              "z0": _u(rng, .05, .09), "size": [_u(rng, .04, .08), _u(rng, .05, .10), .03],
                              "rgb": [_u(rng, .08, .25)] * 3})

    # sensor
    if nominal:
        sensor = {"rgb_gain": 1.0, "rgb_noise": 1.5, "blur": 0.0, "depth_noise_mm": 0.5, "depth_noise_rel": 0.001,
                  "hole_frac": 0.0, "edge_invalid_p": 0.2, "stereo_band_px": 2, "depth_shift_px": [0, 0]}
    elif not unseen:
        sensor = {"rgb_gain": _u(rng, .9, 1.1), "rgb_noise": 3.0, "blur": 0.0, "depth_noise_mm": 1.0,
                  "depth_noise_rel": 0.002, "hole_frac": 0.01, "edge_invalid_p": _u(rng, .1, .4),
                  "stereo_band_px": int(rng.integers(1, 4)), "depth_shift_px": [0, 0]}
    else:
        sensor = {"rgb_gain": _u(rng, .75, 1.25), "rgb_noise": _u(rng, 2, 6), "blur": _u(rng, 0, 1.0),
                  "depth_noise_mm": _u(rng, 1, 2.5), "depth_noise_rel": _u(rng, .002, .004),
                  "hole_frac": _u(rng, .01, .08), "edge_invalid_p": _u(rng, .2, .7),
                  "stereo_band_px": int(rng.integers(2, 7)),
                  "depth_shift_px": [int(rng.integers(-2, 3)), int(rng.integers(-1, 2))]}

    # camera pose error (rendered pose = nominal pose perturbed; perception keeps the nominal calibration)
    jm, jd = (0.0, 0.0) if nominal else (0.01, 1.0)
    cam = {"dt": [_u(rng, -jm, jm) for _ in range(3)], "drot_deg": [_u(rng, -jd, jd) for _ in range(3)]}
    return {"seed": int(seed), "family": family, "layout_mode": layout_mode, "layout": layout, "objects": objects,
            "light": light, "texture": tex, "stickers": stickers, "occluders": occluders, "sensor": sensor,
            "camera": cam, "noise_seed": int(rng.integers(1 << 31))}


# ----------------------------------------------------------------------------------------------- renderer
class SynthRenderer:
    """Ray caster for the head camera. Geometry from scene.yaml, camera model from calibration.yaml (pinhole, no
    distortion: the D435 colour stream is rectified)."""

    def __init__(self, scene: Scene | None = None, cal: Calibration | None = None, cam: str = "head",
                 supersample: bool = True):
        self.sc = scene or Scene.load()
        self.cal = cal or Calibration.load()
        self.cam = cam
        self.K = self.cal.cams[cam].K
        self.T_nominal = self.cal._T(cam)
        self.supersample = supersample
        import cv2
        tex = cv2.imread(str(ASSETS / self.sc.cfg["counter"]["mat_texture"]))
        self.mat_tex = None if tex is None else cv2.cvtColor(tex, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # --------------------------------------------------------------- scene assembly
    def object_planes(self, ob: dict) -> tuple[np.ndarray, np.ndarray, float]:
        """Half-spaces of one layout object; returns (planes, centre (pelvis), pelvis yaw)."""
        sc = self.sc
        shape = ob.get("shape", "cube")
        h = sc.object_height(shape)
        centre = sc.mat_to_pelvis([ob["x"], ob["y"], h / 2])
        yaw = -np.pi / 2 - float(ob["yaw"])              # layout (mat) yaw -> pelvis yaw (layout.mat_yaw_to_pelvis)
        poly = outline(sc, shape, yaw) + centre[:2]
        return prism_planes(poly, sc.counter_z, sc.counter_z + h), centre, yaw

    def solids(self, spec: dict) -> list[dict]:
        """Every solid of the frame: {planes, corners (8,3) for the image ROI, albedo (3,), sheen, tag}."""
        sc, b = self.sc, self.sc.cfg["bins"]
        out = []
        ox, oy = np.asarray(b["outer_size"], float) / 2
        wt, wh, ft = b["wall_thickness"], b["wall_height"], b["floor_thickness"]
        z0 = sc.counter_z
        for name, (cx, cy) in b["centers"].items():
            c = sc.mat_to_pelvis([cx, cy, 0.0])
            # pelvis x = mat y (depth of the box), pelvis y = mat x (width)
            parts = [((0, 0, ft / 2), (oy, ox, ft / 2)),
                     ((oy - wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)), ((-oy + wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)),
                     ((0, ox - wt / 2, wh / 2), (oy, wt / 2, wh / 2)), ((0, -ox + wt / 2, wh / 2), (oy, wt / 2, wh / 2))]
            for off, half in parts:
                ctr = c + np.asarray(off) + [0, 0, 0]
                ctr[2] = z0 + off[2]
                out.append({"planes": box_planes(ctr, half), "corners": _box_corners(ctr, half),
                            "albedo": np.asarray(spec["texture"]["bin_rgb"]), "sheen": 0.1, "tag": f"bin:{name}"})
        for i, ob in enumerate(spec["objects"]):
            planes, centre, yaw = self.object_planes(ob)
            h = sc.object_height(ob.get("shape"))
            R = sc.object_dims(ob.get("shape", "cube"))["circumradius"]
            out.append({"planes": planes, "corners": _box_corners(centre, [R, R, h / 2]),
                        "albedo": np.asarray(ob["albedo"]), "sheen": spec["light"]["sheen"], "tag": f"obj:{i}"})
        for k, oc in enumerate(spec["occluders"]):
            half = np.asarray(oc["size"], float) / 2
            ctr = sc.mat_to_pelvis([oc["x"], oc["y"], oc["z0"] + half[2]])
            ctr = ctr.copy()
            out.append({"planes": box_planes(ctr, half[[1, 0, 2]]), "corners": _box_corners(ctr, half[[1, 0, 2]]),
                        "albedo": np.asarray(oc["rgb"]), "sheen": 0.05, "tag": f"occ:{k}"})
        return out

    # --------------------------------------------------------------- one ray-casting pass
    def _camera(self, spec: dict):
        """Rendered (true) camera pose = nominal pose with the spec's rotation / translation error applied in the
        optical frame. Returns (R pelvis<-optical, origin)."""
        T = self.T_nominal
        dR = axis_angle(np.radians(spec["camera"]["drot_deg"]))
        R = T[:3, :3] @ dR
        o = T[:3, 3] + np.asarray(spec["camera"]["dt"], float)
        return R, o

    def _cast(self, R, o, solids, du, dv):
        """Cast one ray per pixel (pixel centre + (du, dv)). Returns t (H,W) optical depth in m, kind (H,W),
        solid index (H,W), entry-plane index (H,W), rays (H,W,3) pelvis frame scaled so that optical z = 1."""
        K = self.K
        u, v = np.meshgrid(np.arange(W_IMG) + du, np.arange(H_IMG) + dv)
        ray_opt = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        rays = ray_opt @ R.T
        t = np.full((H_IMG, W_IMG), np.inf)
        kind = np.zeros((H_IMG, W_IMG), np.int8)
        sid = np.full((H_IMG, W_IMG), -1, np.int32)
        pid = np.zeros((H_IMG, W_IMG), np.int16)
        rz = rays[..., 2]
        for z, k in ((FLOOR_Z, KIND_FLOOR), (self.sc.counter_z, KIND_COUNTER)):
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = (z - o[2]) / rz
            ok = (rz < -1e-9) & (tp > 0)
            if k == KIND_COUNTER:
                p = o + tp[..., None] * rays
                pm = self.sc.pelvis_to_mat(np.where(ok[..., None], p, 0.0))
                c = self.sc.cfg["counter"]
                mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
                lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
                ok &= np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)
            upd = ok & (tp < t)
            t[upd], kind[upd] = tp[upd], k
        for i, s in enumerate(solids):
            win = self._roi(R, o, s["corners"])
            if win is None:
                continue
            r = rays[win].reshape(-1, 3)
            ts, ks = intersect_convex(o, r, s["planes"])
            ts, ks = ts.reshape(t[win].shape), ks.reshape(t[win].shape)
            sub_t = t[win]
            upd = ts < sub_t
            sub_t[upd] = ts[upd]
            t[win] = sub_t
            for arr, val in ((kind, KIND_SOLID), (sid, i), (pid, ks)):
                sub = arr[win]
                sub[upd] = val if np.isscalar(val) else val[upd]
                arr[win] = sub
        return t, kind, sid, pid, rays

    def _roi(self, R, o, corners):
        """Image window (slice pair) covering the projection of `corners` from camera (R, o); None if not visible."""
        p = (np.asarray(corners) - o) @ R
        if np.any(p[:, 2] <= 0.01):
            return None
        u = self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2]
        u0, u1 = max(int(np.floor(u.min())) - 2, 0), min(int(np.ceil(u.max())) + 3, W_IMG)
        v0, v1 = max(int(np.floor(v.min())) - 2, 0), min(int(np.ceil(v.max())) + 3, H_IMG)
        if u0 >= u1 or v0 >= v1:
            return None
        return slice(v0, v1), slice(u0, u1)

    # --------------------------------------------------------------- shading
    def _texture(self, spec, pm):
        """Albedo (N,3) of counter points given in mat coords (N,3)."""
        tx, sc = spec["texture"], self.sc
        W, D = sc.cfg["counter"]["mat_size"]
        x, y = pm[:, 0], pm[:, 1]
        on_mat = (x >= 0) & (x <= W) & (y >= 0) & (y <= D)
        out = np.tile(np.asarray(tx["counter_rgb"], np.float32), (len(x), 1))
        kind = tx["kind"]
        if kind in ("real", "real_tinted") and self.mat_tex is not None:
            row = np.clip(((D - y) * 1000).astype(int), 0, self.mat_tex.shape[0] - 1)
            col = np.clip(((W - x) * 1000).astype(int), 0, self.mat_tex.shape[1] - 1)
            a = self.mat_tex[row, col]
            if kind == "real":
                a = a * tx["scale"] * np.asarray(tx["rgb_jit"]) + tx["bias"]
            else:                                        # unseen: the real print strongly re-tinted
                a = a * np.asarray(tx["rgb_a"]) / max(np.mean(tx["rgb_a"]), 1e-3) * 0.8
            out[on_mat] = a[on_mat]
            return np.clip(out, 0, 1)
        A, B, per = np.asarray(tx["rgb_a"]), np.asarray(tx["rgb_b"]), tx["period"]
        r = np.random.default_rng(tx["seed"])
        if kind == "solid":
            a = np.tile(A, (len(x), 1))
        elif kind == "checker":
            m = ((np.floor(x / per) + np.floor(y / per)) % 2)[:, None]
            a = A * (1 - m) + B * m
        elif kind == "wood":                             # stripes along x with a wobble: brown / orange grain
            ph = r.uniform(0, 2 * np.pi, 3)
            w = np.sin(y * 2 * np.pi / (per * 0.4) + 3 * np.sin(x * 9 + ph[0]) + ph[1]) * 0.5 + 0.5
            a = A * (1 - w[:, None]) + B * w[:, None]
        else:                                            # blotch: smooth random field of two colours
            k = r.normal(size=(6, 4))
            f = sum(np.sin(x * k[i, 0] * 40 + y * k[i, 1] * 40 + k[i, 2] * 3) for i in range(6)) / 6
            m = (f > 0)[:, None].astype(float)
            a = A * (1 - m) + B * m
        out[on_mat] = a[on_mat]
        return np.clip(out, 0, 1)

    def _shade(self, spec, solids, R, o, t, kind, sid, pid, rays):
        """Linear RGB (H,W,3) in [0, 1+] of one pass."""
        L = spec["light"]
        img = np.zeros((H_IMG, W_IMG, 3), np.float32)
        P = o + np.where(np.isfinite(t), t, 0.0)[..., None] * rays
        n = np.zeros((H_IMG, W_IMG, 3))
        albedo = np.zeros((H_IMG, W_IMG, 3))
        sheen = np.zeros((H_IMG, W_IMG))
        floor = kind == KIND_FLOOR
        albedo[floor] = 0.35
        n[floor] = (0, 0, 1)
        cnt = kind == KIND_COUNTER
        n[cnt] = (0, 0, 1)
        albedo[cnt] = self._texture(spec, self.sc.pelvis_to_mat(P[cnt]))
        sticker_mask = np.zeros((H_IMG, W_IMG), bool)
        if spec["stickers"] and cnt.any():               # flat coloured shapes printed / lying on the mat
            pm = self.sc.pelvis_to_mat(P[cnt])
            alb = albedo[cnt]
            hit_any = np.zeros(len(pm), bool)
            for s in spec["stickers"]:
                poly = shape_polygon(s["shape"], s["size"] / 2, s["yaw"]) + [s["x"], s["y"]]
                e = np.roll(poly, -1, 0) - poly
                inside = np.ones(len(pm), bool)
                for j in range(len(poly)):
                    d = pm[:, :2] - poly[j]
                    inside &= e[j, 0] * d[:, 1] - e[j, 1] * d[:, 0] >= 0
                alb[inside] = s["albedo"]
                hit_any |= inside
            albedo[cnt] = alb
            sticker_mask[cnt] = hit_any
        sol = kind == KIND_SOLID
        if sol.any():
            for i in np.unique(sid[sol]):
                m = sol & (sid == i)
                s = solids[i]
                n[m] = s["planes"][pid[m], :3]
                albedo[m] = s["albedo"]
                sheen[m] = s["sheen"]
        lp = np.asarray(L["key_pos"], float)
        ldir = lp - P
        ldist = np.linalg.norm(ldir, axis=-1, keepdims=True)
        ldir = ldir / np.maximum(ldist, 1e-9)
        lam = np.clip(np.einsum("hwc,hwc->hw", n, ldir), 0, None)
        vis = self._shadow(spec, solids, P, cnt | sol, lp)
        view = -rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        half = ldir + view
        half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-9)
        spec_term = np.clip(np.einsum("hwc,hwc->hw", n, half), 0, None) ** 8
        key_rgb, amb_rgb = np.asarray(L["key_rgb"]), np.asarray(L["ambient_rgb"])
        E = L["ambient"] * amb_rgb + (L["key"] * vis * lam)[..., None] * key_rgb
        img = albedo * E + (sheen * L["key"] * vis * spec_term)[..., None] * key_rgb
        img[kind == KIND_NONE] = 0.02
        return img, sticker_mask

    def _shadow(self, spec, solids, P, mask, lp):
        """1 where the key light reaches the surface point, 0 in the shadow of an object or occluder. Only points
        near an object footprint are tested (shadows of 45 mm objects are short)."""
        vis = np.ones(mask.shape)
        casters = [s for s in solids if not s["tag"].startswith("bin")]
        if not casters:
            return vis
        idx = np.flatnonzero(mask.ravel())
        pts = P.reshape(-1, 3)[idx]
        reach = np.zeros(len(idx), bool)
        for s in casters:
            c = s["corners"].mean(0)
            reach |= np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < 0.25
        idx, pts = idx[reach], pts[reach]
        blocked = np.zeros(len(idx), bool)
        dirs = lp - pts
        for s in casters:
            n, d = s["planes"][:, :3], s["planes"][:, 3]
            # segment test p + t*dir, t in (eps, 1): same clipping as intersect_convex but with per-point origins
            denom = dirs @ n.T
            num = d[None, :] - pts @ n.T
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = num / denom
            t_in = np.where(denom < 0, tt, -np.inf).max(1)
            t_out = np.where(denom > 0, tt, np.inf).min(1)
            outside_par = ((denom == 0) & (num < 0)).any(1)
            blocked |= (t_in <= t_out) & (t_out > 1e-4) & (t_in < 1.0) & ~outside_par & (t_in > -1e9)
        flat = vis.ravel()
        flat[idx[blocked]] = 0.0
        return flat.reshape(mask.shape)

    # --------------------------------------------------------------- full frame
    def render(self, spec: dict):
        """Render one frame. Returns (rgb uint8 (480,640,3), depth uint16 mm (480,640), ground truth dict)."""
        import cv2
        R, o = self._camera(spec)
        solids = self.solids(spec)
        rng = np.random.default_rng(spec["noise_seed"])
        offsets = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] if self.supersample else [(0.0, 0.0)]
        acc = np.zeros((H_IMG, W_IMG, 3), np.float32)
        for du, dv in offsets:
            t, kind, sid, pid, rays = self._cast(R, o, solids, du, dv)
            img, _ = self._shade(spec, solids, R, o, t, kind, sid, pid, rays)
            acc += img
        acc /= len(offsets)
        # centre pass: depth + per-pixel labels for ground truth
        t, kind, sid, pid, rays = self._cast(R, o, solids, 0.0, 0.0)
        S = spec["sensor"]
        if S["blur"] > 0:
            acc = cv2.GaussianBlur(acc, (0, 0), S["blur"])
        rgb = acc * 255.0 * S["rgb_gain"] + rng.normal(0, S["rgb_noise"], acc.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth = np.where(np.isfinite(t), t * 1000.0, 0.0)
        depth = self._depth_sensor(depth, S, rng)
        gt = self._ground_truth(spec, solids, R, o, sid, kind)
        return rgb, depth, gt

    def _depth_sensor(self, depth_mm, S, rng):
        """D435-like depth artefacts on a perfect depth map (mm, 0 = none)."""
        d = depth_mm.copy()
        valid = d > 0
        d = d + rng.normal(0, 1, d.shape) * (S["depth_noise_mm"] + S["depth_noise_rel"] * d)
        # invalid near depth edges (mixed pixels the stereo matcher rejects), with probability edge_invalid_p
        pad = np.pad(depth_mm, 1, mode="edge")
        grad = np.max(np.abs(np.stack([pad[1:-1, 2:], pad[1:-1, :-2], pad[2:, 1:-1], pad[:-2, 1:-1]]) - depth_mm), 0)
        edge = (grad > 8.0) & (rng.random(d.shape) < S["edge_invalid_p"])
        # stereo occlusion band: pixels just LEFT of a nearer surface are not seen by the second imager
        band = np.zeros(d.shape, bool)
        for k in range(1, int(S["stereo_band_px"]) + 1):
            right = np.pad(depth_mm[:, k:], ((0, 0), (0, k)), mode="edge")
            band |= (right > 0) & (depth_mm - right > 20.0)
        holes = rng.random(d.shape) < S["hole_frac"]
        d[~valid | edge | band | holes] = 0
        dx, dy = S["depth_shift_px"]
        if dx or dy:                                     # RGB-depth misregistration
            d = np.roll(np.roll(d, dy, axis=0), dx, axis=1)
        return np.clip(np.round(d), 0, 65535).astype(np.uint16)

    def _ground_truth(self, spec, solids, R, o, sid, kind):
        sc = self.sc
        objs = []
        for i, ob in enumerate(spec["objects"]):
            planes, centre, yaw = self.object_planes(ob)
            j = next(k for k, s in enumerate(solids) if s["tag"] == f"obj:{i}")
            visible = int(((kind == KIND_SOLID) & (sid == j)).sum())
            unocc = self._unoccluded_px(R, o, solids[j])
            h = sc.object_height(ob.get("shape"))
            corners = solids[j]["corners"]
            p = (corners - o) @ R
            u = self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2]
            v = self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2]
            truncated = bool(np.any(p[:, 2] <= 0) or u.min() < 0 or v.min() < 0 or u.max() >= W_IMG or v.max() >= H_IMG)
            objs.append({"color": ob["color"], "shape": ob.get("shape", "cube"), "truncated": truncated,
                         "center": [round(float(v), 6) for v in centre],
                         "top": [round(float(centre[0]), 6), round(float(centre[1]), 6), round(float(centre[2] + h / 2), 6)],
                         "yaw": round(float(yaw), 6), "visible_px": visible, "unoccluded_px": unocc})
        return {"seed": spec["seed"], "family": spec["family"], "layout_mode": spec["layout_mode"], "objects": objs,
                "n_stickers": len(spec["stickers"]), "n_occluders": len(spec["occluders"]),
                "camera_error": spec["camera"], "light_style": spec["light"].get("style", spec["family"]),
                "texture": spec["texture"]["kind"]}

    def _unoccluded_px(self, R, o, solid):
        """Pixels the object would cover if nothing else were in front of it (for 'how occluded' in the GT)."""
        win = self._roi(R, o, solid["corners"])
        if win is None:
            return 0
        K = self.K
        u, v = np.meshgrid(np.arange(win[1].start, win[1].stop) + 0.0, np.arange(win[0].start, win[0].stop) + 0.0)
        ray_opt = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        ts, _ = intersect_convex(o, (ray_opt @ R.T).reshape(-1, 3), solid["planes"])
        return int(np.isfinite(ts).sum())


def outline(sc: Scene, shape: str, yaw: float) -> np.ndarray:
    """Top-view outline (N,2, CCW, centred on the centroid) of an object in the pelvis frame at pelvis yaw `yaw`.
    Sharp corners (the 2.5 mm fillets are ignored: < 1 px at the head camera's range)."""
    a = sc.object_dims(shape)["apothem"]
    if shape == "cylinder":
        ang = yaw + 2 * np.pi * np.arange(CYLINDER_SIDES) / CYLINDER_SIDES
        # vertex radius halfway between "polygon inside the circle" and "polygon around it": error < 0.03 mm
        r = 0.5 * (a + a / np.cos(np.pi / CYLINDER_SIDES))
        return r * np.stack([np.cos(ang), np.sin(ang)], -1)
    return shape_polygon(shape, a, yaw)


def _box_corners(c, half):
    c, h = np.asarray(c, float), np.asarray(half, float)
    return np.array([c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def frame_hash(rgb: np.ndarray, depth: np.ndarray) -> str:
    """SHA-256 (first 16 hex) of the frame bytes: frozen manifests store it to prove a frame is re-generated exactly."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rgb).tobytes())
    h.update(np.ascontiguousarray(depth).tobytes())
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", choices=FAMILIES, default="train")
    ap.add_argument("--layout-mode", choices=LAYOUT_MODES, default="band")
    ap.add_argument("--out", default=None, help="write RGB | depth side by side (png)")
    a = ap.parse_args(argv)
    import cv2
    r = SynthRenderer()
    spec = sample_spec(a.seed, a.family, a.layout_mode)
    rgb, depth, gt = r.render(spec)
    for o in gt["objects"]:
        print(o)
    print("light:", spec["light"].get("style", a.family), "texture:", spec["texture"]["kind"],
          "stickers:", len(spec["stickers"]), "occluders:", len(spec["occluders"]), "hash:", frame_hash(rgb, depth))
    if a.out:
        dv = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / 1000.0), cv2.COLORMAP_JET)
        dv[depth == 0] = 0
        cv2.imwrite(a.out, np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dv]))
        print("->", a.out)


if __name__ == "__main__":
    main()
