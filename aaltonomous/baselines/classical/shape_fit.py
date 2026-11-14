"""Shape, yaw and centre of one object from the head camera, using the KNOWN object geometry (scene.yaml).

Two geometry-aware methods (both used by perception.py; compared in loop/vision, see loop/STATE.md):

(b) TopFaceFitter: polygon fit on the top-face points.
    The depth pixels of the object's top face, deprojected to the pelvis frame, are a perspective-free top view of
    the top face. They are rasterised on a metric grid (1 mm cells) around their centroid; that footprint F is
    compared (intersection over union) with the exact top outline of each shape (cube square, cylinder circle,
    triangle) at every yaw, using templates precomputed once. Best IoU -> shape; its yaw and a small xy offset
    search -> yaw and centre. Needs depth on the top face.

(a) SilhouetteFitter: render-and-compare.
    For a hypothesis (shape, x, y, yaw) the whole 3D prism standing on the counter plane is projected into the
    image with the calibrated camera: the image of a convex prism is the convex hull of its projected vertices, so
    the silhouette is one cv2.fillConvexPoly. It is scored by IoU against the colour blob. xy is refined by moving
    the hypothesis until the silhouette centroid matches the blob centroid (2x2 Jacobian of the projection), yaw by
    a coarse-to-fine search. Works from RGB alone (no depth needed), so it is the fallback for depth holes.

Neither method uses 2D image templates of objects: every hypothesis is rendered from the true 3D geometry at its
place on the counter, so perspective is handled exactly.

Conventions: pelvis frame (x forward, y left, z up), metres, radians. `yaw` = rotation about +z of the canonical
outline scene.shape_polygon(shape) (square with axis-aligned sides; triangle apex towards +y). Symmetry periods:
cube pi/2, triangle 2 pi/3; returned yaws are wrapped to [-period/2, period/2). Cylinder: yaw None.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltonomous.scene.scene import SHAPES, shape_polygon

PERIOD = {"cube": np.pi / 2, "triangle": 2 * np.pi / 3, "cylinder": None}
CYLINDER_SIDES = 48


def outline(scene, shape: str, yaw: float = 0.0) -> np.ndarray:
    """Top outline (N,2), CCW, centred on the centroid, of `shape` at yaw `yaw` (pelvis xy, metres).
    Sharp corners: the 2.5 mm edge fillets are below the camera resolution at 0.5 m (~0.8 mm / px)."""
    a = scene.object_dims(shape)["apothem"]
    if shape == "cylinder":
        ang = yaw + 2 * np.pi * np.arange(CYLINDER_SIDES) / CYLINDER_SIDES
        r = 0.5 * (a + a / np.cos(np.pi / CYLINDER_SIDES))
        return r * np.stack([np.cos(ang), np.sin(ang)], -1)
    return shape_polygon(shape, a, yaw)


def wrap_yaw(yaw: float | None, shape: str | None) -> float | None:
    """Wrap a yaw into [-period/2, period/2) of the shape's symmetry; None for cylinders / unknown shapes."""
    if yaw is None or shape is None or PERIOD.get(shape) is None:
        return None
    p = PERIOD[shape]
    return float((yaw + p / 2) % p - p / 2)


@dataclass
class Fit:
    """Result of one fitter for one blob.

    shape      best shape, or None when the evidence is not clear enough (see `margin`)
    yaw        rad, wrapped to the shape's symmetry (None for cylinder / unknown)
    xy         fitted centre (pelvis, m); None if the fitter could not run
    iou        {shape: best IoU in [0, 1]}
    margin     best IoU - second best IoU (confidence of the shape decision)
    """
    shape: str | None
    yaw: float | None
    xy: np.ndarray | None
    iou: dict = field(default_factory=dict)
    margin: float = 0.0
    method: str = ""
    yaws: dict = field(default_factory=dict)    # {shape: best yaw of that shape} (for combining fitters)
    xys: dict = field(default_factory=dict)     # {shape: best centre of that shape}
    vis: float = 1.0                            # polygon fit: observable share of the best outline (1 = unhidden)


def decide(iou: dict, min_iou: float, min_margin: float):
    """Pick the shape with the highest IoU if it is good enough and clearly better than the runner-up."""
    if not iou:
        return None, 0.0
    order = sorted(iou, key=iou.get, reverse=True)
    best = order[0]
    margin = iou[best] - (iou[order[1]] if len(order) > 1 else 0.0)
    if iou[best] < min_iou or margin < min_margin:
        return None, margin
    return best, margin


# ------------------------------------------------------------------------------------------- (b) top-face polygon fit
class TopFaceFitter:
    """Polygon fit of the top-face footprint on a metric grid. Templates are built once per scene geometry.

    res_m      grid cell (m). 1 mm: a head-camera pixel covers ~0.8-1.2 mm of the counter at 0.4-0.6 m range.
    half_m     half-width of the grid (m): must hold the largest outline (cube corner 31.8 mm) + centring error.
    yaw_step   template yaw step (deg); the result is refined by parabolic interpolation between neighbours.
    close_px   morphological closing (cells) that fills the gaps between deprojected pixels.
    """

    def __init__(self, scene, res_m=0.001, half_m=0.040, yaw_step_deg=2.0, close_px=5):
        import cv2
        self.res, self.G = res_m, int(round(2 * half_m / res_m))
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)) if close_px > 1 else None
        self.templates, self.flat32 = {}, {}
        for s in SHAPES:
            period = PERIOD[s]
            yaws = np.array([0.0]) if period is None else np.radians(np.arange(0, np.degrees(period), yaw_step_deg))
            stack = np.stack([self._raster_poly(outline(scene, s, y)) for y in yaws]).astype(bool)
            self.templates[s] = (yaws, stack, stack.reshape(len(yaws), -1).sum(1))
            # float32 copy: |T & F| for every template = one matrix-vector product
            self.flat32[s] = stack.reshape(len(yaws), -1).astype(np.float32)
        self.step = np.radians(yaw_step_deg)

    def _cells(self, xy):
        """Metric offsets from the grid centre (m) -> fractional cell coordinates (col = x, row = y)."""
        return np.asarray(xy) / self.res + self.G / 2

    def _raster_poly(self, poly_xy):
        import cv2
        img = np.zeros((self.G, self.G), np.uint8)
        pts = np.round(self._cells(poly_xy) * 16 - 8).astype(np.int32)     # 4 fractional bits, cell centres at +.5
        cv2.fillConvexPoly(img, pts, 1, lineType=cv2.LINE_8, shift=4)
        return img

    def footprint(self, xy_pts: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Binary top-face footprint (G,G) of points (N,2) around `centre`."""
        import cv2
        c = np.floor(self._cells(xy_pts - centre)).astype(int)
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.G) & (c[:, 1] >= 0) & (c[:, 1] < self.G)
        F = np.zeros((self.G, self.G), np.uint8)
        F[c[ok, 1], c[ok, 0]] = 1
        if self.close_k is not None:
            F = cv2.morphologyEx(F, cv2.MORPH_CLOSE, self.close_k)
        return F

    def split(self, xy_pts: np.ndarray, min_points: int = 20, cell_m: float = 0.002) -> list:
        """Group top-face points (N,2) into separate objects: connected components of their footprint on a 2 mm
        grid (closed by one cell, so the gaps between deprojected pixels do not split an object). Groups with fewer
        than `min_points` points are dropped unless nothing else is left. Returns a list of index arrays."""
        import cv2
        lo = xy_pts.min(0)
        c = np.floor((xy_pts - lo) / cell_m).astype(int)
        img = np.zeros((c[:, 1].max() + 3, c[:, 0].max() + 3), np.uint8)
        img[c[:, 1] + 1, c[:, 0] + 1] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(img, connectivity=8)
        if n <= 2:
            return [np.arange(len(xy_pts))]
        k = lab[c[:, 1] + 1, c[:, 0] + 1]
        groups = [np.flatnonzero(k == j) for j in range(1, n)]
        big = [g for g in groups if len(g) >= min_points]
        return big or [np.arange(len(xy_pts))]

    def cell_centres(self, centre: np.ndarray) -> np.ndarray:
        """Pelvis xy (G*G, 2) of the grid cell centres around `centre`, row-major (row = y, col = x)."""
        g = (np.arange(self.G) + 0.5 - self.G / 2) * self.res
        X, Y = np.meshgrid(g, g)
        return np.stack([X.ravel(), Y.ravel()], -1) + centre

    def split_neck(self, xy_pts: np.ndarray, max_extent: float, min_points: int = 20, bin_m: float = 0.001,
                   edge_m: float = 0.015) -> list:
        """Two objects whose top faces nearly touch (same colour, < ~1 cm apart) can be ONE connected footprint:
        mixed depth at the gap between two tops of the same height fills it at top height. One object never
        spans more than `max_extent` (cube diagonal); if the points do (2nd-98th percentile along the principal
        axis), cut at the least dense 1 mm slice (smoothed over 5 mm) at least `edge_m` from either end.
        Returns [idx] (no split) or [idx_a, idx_b]."""
        if len(xy_pts) < 2 * min_points:
            return [np.arange(len(xy_pts))]
        c = xy_pts.mean(0)
        _, _, vt = np.linalg.svd(xy_pts - c, full_matrices=False)
        t = (xy_pts - c) @ vt[0]
        lo, hi = np.percentile(t, [2, 98])
        if hi - lo <= max_extent:
            return [np.arange(len(xy_pts))]
        edges = np.arange(lo, hi + bin_m, bin_m)
        h, _ = np.histogram(t, edges)
        hs = np.convolve(h, np.ones(5) / 5, mode="same")
        k0, k1 = int(edge_m / bin_m), len(hs) - int(edge_m / bin_m)
        if k1 <= k0:
            return [np.arange(len(xy_pts))]
        k = k0 + int(np.argmin(hs[k0:k1]))
        cut = edges[k] + bin_m / 2
        a, b = np.flatnonzero(t < cut), np.flatnonzero(t >= cut)
        if len(a) < min_points or len(b) < min_points:
            return [np.arange(len(xy_pts))]
        return [a, b]

    def fit(self, xy_pts: np.ndarray, min_iou=0.55, min_margin=0.04, shift_mm=2, observe=None,
            min_known=0.6, hidden_vis=1.0, hidden_margin=None, hidden_min_iou=None) -> Fit:
        """Fit every shape to the top-face points (N,2, pelvis xy, m). Returns the best Fit.

        observe: optional callback cells (M,2) -> (occupied (M,) bool, known (M,) bool). With it, the footprint is
        built by looking at every grid cell in the depth image instead of rasterising the points: a cell is
        occupied if the camera sees the top surface there, empty if it sees something farther (the counter), and
        UNKNOWN if it has no depth or is hidden behind something nearer (an occluder, a hand). Unknown cells are
        left out of the IoU, so a partly hidden object is not mistaken for a smaller / other shape. If less than
        `min_known` of the best outline is visible, the shape is not decided (None)."""
        if len(xy_pts) < 20:
            return Fit(None, None, None, method="polygon")
        import cv2
        c0 = xy_pts.mean(0)
        if observe is None:
            F = self.footprint(xy_pts, c0)
            K = np.ones_like(F)
        else:
            occ, known = observe(self.cell_centres(c0))
            F = occ.reshape(self.G, self.G).astype(np.uint8)
            K = (known | occ).reshape(self.G, self.G).astype(np.uint8)
        nF = int(np.count_nonzero(F))
        if nF < 20:
            return Fit(None, None, None, method="polygon")
        flatF, flatK = F.ravel().astype(np.float32), K.ravel().astype(np.float32)
        # padded by the shift range: matchTemplate(Fp, T) = |T & shifted F|, matchTemplate(Kp, T) = |T & shifted K|
        k = shift_mm
        Fp, Kp = np.pad(F, k).astype(np.float32), np.pad(K, k).astype(np.float32)
        best = {}
        for s, (yaws, stack, areas) in self.templates.items():
            flat = self.flat32[s]
            inter = flat @ flatF
            tk = flat @ flatK                                      # template cells that are observable
            iou = inter / np.maximum(tk + nF - inter, 1)
            kb = int(iou.argmax())
            best_iou, best_off, best_k, best_vis = iou[kb], (0, 0), kb, tk[kb] / areas[kb]
            for kk in (kb,):                                       # xy search on the best yaw only
                T = stack[kk].astype(np.float32)
                I = cv2.matchTemplate(Fp, T, cv2.TM_CCORR)              # (2k+1, 2k+1)
                IK = cv2.matchTemplate(Kp, T, cv2.TM_CCORR)
                V = I / np.maximum(IK + nF - I, 1)
                iy, ix = np.unravel_index(int(V.argmax()), V.shape)
                if V[iy, ix] > best_iou + 1e-9:
                    # result (iy, ix) = template placed at offset (ix - k, iy - k) in F = F shifted by (k - ix, k - iy)
                    best_iou, best_off, best_k = float(V[iy, ix]), (k - ix, k - iy), kk
                    best_vis = float(IK[iy, ix]) / areas[kk]
            yaw = None
            if len(yaws) > 1:
                # parabolic refinement of the yaw between neighbouring templates (at the chosen offset)
                Fs = np.roll(np.roll(F, best_off[1], 0), best_off[0], 1).ravel().astype(bool)
                Ks = np.roll(np.roll(K, best_off[1], 0), best_off[0], 1).ravel().astype(bool)
                vals = []
                for d in (-1, 0, 1):
                    t = stack[(best_k + d) % len(yaws)].ravel()
                    i = np.count_nonzero(t & Fs)
                    vals.append(i / max(np.count_nonzero(t & Ks) + nF - i, 1))
                den = vals[0] - 2 * vals[1] + vals[2]
                frac = 0.5 * (vals[0] - vals[2]) / den if den < 0 else 0.0
                yaw = wrap_yaw(yaws[best_k] + np.clip(frac, -0.5, 0.5) * self.step, s)
            # the footprint was shifted by +off cells to match a template centred on c0 -> object centre = c0 - off
            xy = c0 - np.array(best_off, float) * self.res
            best[s] = (float(best_iou), yaw, xy, best_vis)
        iou = {s: v[0] for s, v in best.items()}
        top = max(iou, key=iou.get)
        vis = float(min(best[top][3], 1.0))
        if max(v[3] for v in best.values()) < min_known:              # too little of the object is visible
            return Fit(None, None, best[top][2], iou, 0.0, "polygon", {s: v[1] for s, v in best.items()},
                       {s: v[2] for s, v in best.items()}, vis)
        # partly hidden (unknown cells inside the outline): a missing part can turn one outline into another,
        # so a clearer margin is needed before a shape is named
        hidden = vis < hidden_vis
        need = hidden_margin if (hidden_margin is not None and hidden) else min_margin
        need_iou = hidden_min_iou if (hidden_min_iou is not None and hidden) else min_iou
        shape, margin = decide(iou, max(min_iou, need_iou), max(min_margin, need))
        return Fit(shape, best[top][1] if shape else None, best[top][2], iou, margin, "polygon",
                   {s: v[1] for s, v in best.items()}, {s: v[2] for s, v in best.items()}, vis)


# ------------------------------------------------------------------------------------------- (a) render-and-compare
class SilhouetteFitter:
    """Render-and-compare of the full prism silhouette against the colour blob, in the image.

    cal, cam   Calibration + camera name (pinhole; the projection uses calibration.yaml K and T_pelvis_optical).
    z0, height counter plane height (pelvis z, m) and object height (m): the prism spans z0 .. z0 + height.
    """

    def __init__(self, scene, cal, cam="head", z0=None, height=None, coarse_deg=10.0):
        self.sc, self.cal, self.cam = scene, cal, cam
        T = cal._T(cam)
        self.R, self.o, self.K = T[:3, :3], T[:3, 3], cal.cams[cam].K
        self.z0 = scene.counter_z if z0 is None else z0
        self.h = scene.object_height() if height is None else height
        self.coarse = np.radians(coarse_deg)
        self.unit = {s: outline(scene, s, 0.0) for s in SHAPES}

    def set_camera(self, R, o):
        """Use a corrected camera pose (e.g. after the counter-plane refinement in perception.py)."""
        self.R, self.o = R, o

    def project(self, P):
        """Pelvis points (N,3) -> pixel coordinates (N,2)."""
        p = (P - self.o) @ self.R
        return np.stack([self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2],
                         self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2]], -1)

    def silhouette_poly(self, shape, xy, yaw):
        """Image polygon (convex hull of the projected prism vertices), pixel coordinates (M,2)."""
        import cv2
        c, s = np.cos(yaw), np.sin(yaw)
        o = self.unit[shape] @ np.array([[c, s], [-s, c]]) + xy        # rotate the canonical outline by yaw
        n = len(o)
        P = np.empty((2 * n, 3))
        P[:n, :2], P[n:, :2] = o, o
        P[:n, 2], P[n:, 2] = self.z0, self.z0 + self.h
        uv = self.project(P).astype(np.float32)
        return cv2.convexHull(uv)[:, 0, :]

    def _score(self, shape, xy, yaw, B, off, nB, canvas):
        import cv2
        poly = self.silhouette_poly(shape, xy, yaw) - off
        canvas[:] = 0
        cv2.fillConvexPoly(canvas, np.round(poly * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        nS = int(np.count_nonzero(canvas))
        inter = int(np.count_nonzero(canvas & B))
        m = cv2.moments(canvas, binaryImage=True)
        cen = (np.array([m["m10"], m["m01"]]) / m["m00"] + off) if m["m00"] > 0 else None
        return inter / max(nS + nB - inter, 1), cen

    def _jacobian(self, xy):
        """d(pixel)/d(xy) of the prism's mid-height point (2x2), numerically (1 mm steps)."""
        zc = self.z0 + self.h / 2
        P = np.array([[xy[0], xy[1], zc], [xy[0] + 1e-3, xy[1], zc], [xy[0], xy[1] + 1e-3, zc]])
        uv = self.project(P)
        return np.stack([(uv[1] - uv[0]) / 1e-3, (uv[2] - uv[0]) / 1e-3], -1)

    def fit(self, B: np.ndarray, off: np.ndarray, xy0: np.ndarray, min_iou=0.6, min_margin=0.03,
            shapes=SHAPES) -> Fit:
        """B: binary blob mask (uint8 0/1) of an image ROI whose top-left pixel is `off` (u, v); xy0 initial
        centre (pelvis xy). Returns the best Fit over `shapes`."""
        import cv2
        nB = int(np.count_nonzero(B))
        if nB < 30:
            return Fit(None, None, None, method="render")
        m = cv2.moments(B, binaryImage=True)
        blob_c = np.array([m["m10"], m["m01"]]) / m["m00"] + off
        canvas = np.zeros_like(B)
        res = {}
        for s in shapes:
            period = PERIOD[s]
            xy = np.asarray(xy0, float).copy()
            yaws = [0.0] if period is None else list(np.arange(0, period, self.coarse))
            # 1) align the centroid at yaw 0 / mid coarse grid, 2) coarse yaw, 3) align again, 4) fine yaw
            for _ in range(2):
                _, cen = self._score(s, xy, yaws[len(yaws) // 2], B, off, nB, canvas)
                if cen is None:
                    break
                xy = xy + np.clip(np.linalg.solve(self._jacobian(xy), blob_c - cen), -0.01, 0.01)
            scores = [self._score(s, xy, y, B, off, nB, canvas)[0] for y in yaws]
            k = int(np.argmax(scores))
            yaw, best = yaws[k], scores[k]
            for _ in range(2):
                _, cen = self._score(s, xy, yaw, B, off, nB, canvas)
                if cen is None:
                    break
                xy = xy + np.clip(np.linalg.solve(self._jacobian(xy), blob_c - cen), -0.01, 0.01)
            best = self._score(s, xy, yaw, B, off, nB, canvas)[0]
            if period is not None:
                step = self.coarse / 2
                while step > np.radians(0.6):
                    for y in (yaw - step, yaw + step):
                        v = self._score(s, xy, y, B, off, nB, canvas)[0]
                        if v > best:
                            best, yaw = v, y
                    step /= 2
            res[s] = (best, wrap_yaw(yaw, s), xy)
        iou = {s: v[0] for s, v in res.items()}
        shape, margin = decide(iou, min_iou, min_margin)
        top = max(iou, key=iou.get)
        return Fit(shape, res[top][1] if shape else None, res[top][2], iou, margin, "render",
                   {s: v[1] for s, v in res.items()}, {s: v[2] for s, v in res.items()})
