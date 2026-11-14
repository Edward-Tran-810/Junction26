"""Head RGB-D -> objects on the counter (colour, shape, top-face centre, yaw) in the pelvis frame.

    per = Perception(cfg, scene, calibration)
    per.reset_calibration()             # at every episode start (ClassicalPolicy.reset does it)
    det = per.detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw, uv, shape, shape_conf), ...]

STATEFUL (v2 with perception.refine_cache): the camera correction estimated on the first frame is reused for the
next refine_cache.every_n frames. Call reset_calibration() whenever the camera may have moved or the frames come
from different scenes / recordings (evaluation scripts do it per frame).

Two pipelines, chosen by config.yaml `perception.pipeline`:

legacy (2026-11-11, kept unchanged for comparison and as a fallback)
    Per colour: HSV threshold -> opening -> connected components. Per blob: erode, deproject the pixels with valid
    depth, keep the top face (points within `top_band` of a robust max height), top centre = footprint-weighted mean.
    Too few depth pixels -> blob centroid ray intersected with a plane near the object top (method "plane").
    Shape-agnostic; yaw = minAreaRect of the top points folded to +-45 deg (square symmetry only).

v2 (vision loop, 2026-11-14; design + measurements in loop/STATE.md)
    1. Depth geometry. Every valid depth pixel is deprojected once with per-pixel rays precomputed from
       calibration.yaml (no per-frame trigonometry). The counter plane is fitted to the depth (robust least
       squares) and the camera pose is corrected so that the plane is level at scene.yaml's counter height
       ("plane refinement": removes camera pitch / roll / height error; yaw and x / y error are not observable from
       a plane). Pixels whose 3D point is at object height, on the counter and outside every bin are "object
       pixels".
    2. Colour. A pixel belongs to a colour if (object pixel AND relaxed HSV range) OR (no depth AND strict HSV
       range, outside the projected bin boxes). Geometry, not colour, rejects the orange bins, a brown / orange /
       red counter texture and flat coloured stickers, so the colour ranges can be wide (robust to lighting).
       Without a depth stream the pipeline falls back to the strict HSV ranges + projected bin hulls (as legacy).
    3. Per blob: top-face points (within `top_band` of the robust top) -> shape fit with the KNOWN geometry
       (shape_fit.py): (b) polygon fit of the top-face footprint and / or (a) render-and-compare of the projected
       3D prism against the colour silhouette. `perception.shape.method`: polygon | render | combined.
       The shape is only reported when the fit is clear (IoU and margin thresholds); otherwise shape = None and
       the policy treats the object as "colour only" (instruction.target_bin), never as a guessed shape.
    4. Same rejection rules as legacy (height, off the mat, inside a bin).

Coordinate frames: pelvis (x forward, y left, z up, metres); image pixels (u right, v down); depth uint16 mm aligned
to the colour image (0 = invalid). Calibration: interface/calibration.yaml (provisional until M7, see STATE.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shape_fit import SilhouetteFitter, TopFaceFitter, decide, wrap_yaw


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int                  # pixels in the colour blob
    n_depth: int               # blob pixels with valid object-height depth
    method: str                # "depth" | "plane" (no usable depth: ray / silhouette on the known counter plane)
    yaw: float | None = None   # rad about pelvis +z of the canonical outline, wrapped to the shape's symmetry
                               # (cube +-45 deg, triangle +-60 deg); None if cylinder or shape unknown.
                               # legacy pipeline: minAreaRect angle folded to +-45 deg (info only).
    uv: tuple = field(default=(0.0, 0.0))   # blob pixel (median u, v), for debugging / overlays
    shape: str | None = None   # "cube" | "cylinder" | "triangle" | None (= not sure; never a guess)
    shape_conf: float = 0.0    # IoU margin between the best and the second-best shape (0..1)
    fit_iou: dict | None = None   # {shape: IoU} of the shape fit (debug)
    visible: float | None = None   # v2 polygon fit: observable share of the fitted outline (< 1 = partly hidden)
    color_conf: float = 1.0    # v2: share of the winning colour in the blob's saturation-weighted vote (0..1).
                               # Below perception.colour.min_conf the colour is reported as None (abstain).

    @property
    def confidence(self) -> float:
        """One 0-1 score for the policy: the weaker of colour and (if a shape is reported) shape confidence."""
        return min(self.color_conf, self.shape_conf) if self.shape is not None else self.color_conf

    def __repr__(self):
        x, y, z = self.top
        sh = f" {self.shape}" if self.shape else ""
        yaw = f" yaw={np.degrees(self.yaw):+.0f}" if self.yaw is not None else ""
        return (f"{str(self.color):<6}{sh} top=({x:+.3f},{y:+.3f},{z:+.3f}){yaw} px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]")


@dataclass
class DepthGeometry:
    """Per-frame 3D view of the depth image (pelvis frame, refined camera).
    Z: (H,W) float32 pelvis z of every pixel with the refined camera (only meaningful where `valid`); d: depth (m).
    X, Y: not kept (None): full-image x / y are never needed, blob points are deprojected on demand.
    valid: depth in range; obj: uint8 255 = object pixel; not_counter: uint8 255 = not confirmed counter surface.
    All (H,W) arrays cover only the object window `win` of the image (Perception._object_roi).
    R, o: refined camera rotation (pelvis <- optical) and origin; plane: refinement report (dict)."""
    valid: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    d: np.ndarray
    obj: np.ndarray
    not_counter: np.ndarray
    R: np.ndarray
    o: np.ndarray
    plane: dict
    win: tuple = (0, 480, 0, 640)   # (v0, v1, u0, u1): the arrays above cover this image window only


class Perception:
    def __init__(self, cfg, scene, calibration):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # pelvis-frame xy correction added to every detection (head extrinsics error; tuned on the robot)
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)
        self.pipeline = self.cfg.get("pipeline", "legacy")
        self.shape_cfg = self.cfg.get("shape") or {}
        self._rays = None
        self._hulls = {}
        self._top_fitter = None
        self._wide_fitter = None
        self.last_wb = None
        self._sil_fitter = None
        self.last_plane = None
        self._cam_cache = None                  # (R, o, report, frames since estimate): see reset_calibration()

    def reset_calibration(self):
        """Forget the cached camera correction (call at the start of every episode). The head camera is fixed on
        the robot, so within an episode the plane / bin refinement is estimated on the first frame with depth and
        reused; it is re-estimated every perception.refine_cache.every_n frames (0 / disabled = every frame)."""
        self._cam_cache = None

    # ================================================================================================ common helpers
    def masks(self, rgb):
        """{colour: uint8 mask} of the STRICT HSV ranges (perception.hsv), after opening. Used by legacy, by the
        no-depth fallback of v2 and by run.py's debug image."""
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        return {color: self._hsv_mask(hsv, ranges) for color, ranges in self.cfg["hsv"].items()}

    def _hsv_mask(self, hsv, ranges, open_px=None):
        import cv2
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        k = int(self.cfg["open_px"]) if open_px is None else open_px
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        return m

    def detect(self, rgb, depth, debug=False):
        """Detections in one head frame. debug=True also returns the rejected blobs [(Detection | None, reason)]."""
        if self.pipeline == "legacy":
            return self._detect_legacy(rgb, depth, debug)
        return self._detect_v2(rgb, depth, debug)

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        if shape in self._hulls:
            return self._hulls[shape]
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        self._hulls[shape] = keep
        return keep

    def _reject(self, det):
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        return None

    # ================================================================================================ v2 pipeline
    def _pixel_rays(self, shape):
        """(3, H, W) float32 ray directions in the pelvis frame for the NOMINAL camera, scaled so that the optical
        z component is 1: a pixel with depth d (m) is at o + d * ray. Computed once per image size."""
        if self._rays is None or self._rays.shape[1:] != shape:
            K = self.cal.cams[self.cam].K
            R = self.cal._T(self.cam)[:3, :3]
            v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
            ray_opt = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)])
            self._rays = np.einsum("ij,jhw->ihw", R, ray_opt).astype(np.float32)
        return self._rays

    def _object_roi(self, shape):
        """(v0, v1, u0, u1): the image window in which an object on the mat in front of the bins can appear
        (projection of that volume, counter to 1.5 object heights, with the nominal camera, + `roi_margin_px` for
        camera error). The full-resolution part of v2 runs only inside it (half the image: below the bins)."""
        key = ("roi", shape)
        if key not in self._hulls:
            sc = self.scene
            W, D = sc.cfg["counter"]["mat_size"]
            near_bins = min(cy for _, cy in sc.cfg["bins"]["centers"].values()) - sc.cfg["bins"]["outer_size"][1] / 2
            pts = np.array([[x, y, z] for x in np.linspace(0, W, 10) for y in (0.0, near_bins + 0.02)
                            for z in (0.0, 1.5 * self.height)])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(pts))
            uvz = uvz[uvz[:, 2] > 0]
            m = int(self.cfg.get("roi_margin_px", 25))
            v0 = int(np.clip(np.floor(uvz[:, 1].min()) - m, 0, shape[0]))
            v1 = int(np.clip(np.ceil(uvz[:, 1].max()) + m, 0, shape[0]))
            u0 = int(np.clip(np.floor(uvz[:, 0].min()) - m, 0, shape[1]))
            u1 = int(np.clip(np.ceil(uvz[:, 0].max()) + m, 0, shape[1]))
            self._hulls[key] = (v0, v1, u0, u1) if v1 > v0 and u1 > u0 else (0, shape[0], 0, shape[1])
        return self._hulls[key]

    def _depth_geometry(self, depth, win=None) -> DepthGeometry | None:
        """Deproject the depth image, refine the camera pose on the counter plane, classify pixels.
        Returns None when the frame has (almost) no depth (stream missing): the caller then uses colour only."""
        if depth is None:
            return None
        c, sc = self.cfg, self.scene
        dlo, dhi = c["depth_range_mm"]
        valid = (depth >= dlo) & (depth <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return None
        rays = self._pixel_rays(depth.shape)
        T = self.cal._T(self.cam)
        R0, o0 = T[:3, :3], T[:3, 3]
        d = depth.astype(np.float32) * np.float32(self.cal.cams[self.cam].depth_scale)
        # camera refinement on a sparse pixel grid (nominal-camera points P = o0 + d * ray); cached per episode
        R, o, plane = R0, o0, {"applied": False}
        cache = self.cfg.get("refine_cache") or {}
        every = int(cache.get("every_n", 0)) if cache.get("enabled", False) else 0
        if every > 0 and self._cam_cache is not None and self._cam_cache[3] < every:
            R, o, plane, n = self._cam_cache
            plane = dict(plane, cached=True)
            self._cam_cache = (R, o, plane, n + 1)
        elif (self.cfg.get("plane_refine") or {}).get("enabled") or (self.cfg.get("bin_refine") or {}).get("enabled"):
            s = int((self.cfg.get("plane_refine") or {}).get("stride", 4))
            Ps = o0 + d[::s, ::s, None] * np.moveaxis(rays[:, ::s, ::s], 0, -1)
            Rc, dz, plane = self._plane_refinement(valid[::s, ::s], Ps, o0)
            R, o = Rc @ R0, o0 + np.array([0.0, 0.0, dz])
            R, o, plane["bins"] = self._bin_refinement(valid[::s, ::s], Ps, R, o, R0, o0)
            if every > 0:
                self._cam_cache = (R, o, plane, 1)
        # dense part: only inside the object window
        v0, v1, u0, u1 = win or (0, depth.shape[0], 0, depth.shape[1])
        rays, d, valid = rays[:, v0:v1, u0:u1], d[v0:v1, u0:u1], valid[v0:v1, u0:u1]
        # height of every pixel with the refined camera: z = o_z + d * (row 3 of R R0^T) . ray_nominal
        rc = (R @ R0.T)[2].astype(np.float32)
        Z = np.float32(o[2]) + d * (rc[0] * rays[0] + rc[1] * rays[1] + rc[2] * rays[2])
        h = (Z - np.float32(sc.counter_z)) * np.float32(1.0 / self.height)
        lo, hi = c.get("object_px_height", [0.2, 1.5])
        cand = valid & (h >= lo) & (h <= hi)
        counter = valid & (h < lo)
        # object-height pixels must lie on the mat and outside every bin (xy of the candidates only: few pixels)
        vv, uu = np.nonzero(cand)
        P = o + (d[vv, uu, None] * rays[:, vv, uu].T) @ (R @ R0.T).T
        pm_x, pm_y = P[:, 1] - sc.origin[1], P[:, 0] - sc.origin[0]   # mat x = pelvis y - origin, mat y = pelvis x
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        ok = (pm_x >= -m) & (pm_x <= W + m) & (pm_y >= -m) & (pm_y <= D + m)
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for bx, by in sc.cfg["bins"]["centers"].values():
            ok &= ~((np.abs(pm_x - bx) <= half[0]) & (np.abs(pm_y - by) <= half[1]))
        obj = np.zeros(valid.shape, np.uint8)
        obj[vv[ok], uu[ok]] = 255
        return DepthGeometry(valid, None, None, Z, d, obj, (~counter).astype(np.uint8) * 255, R, o, plane,
                             (v0, v1, u0, u1))

    def _plane_refinement(self, valid, P, o):
        """Fit the counter plane z = a x + b y + c to depth points near the nominal counter height (P: (h, w, 3)
        nominal-camera points of a sparse pixel grid, valid: (h, w); three rounds of trimmed least squares) and return the camera correction (Rc, dz, report):
        Rc rotates the measured plane normal onto +z about the camera centre, dz then moves the plane to
        scene.counter_z. Applied only within the configured limits (else identity, report['applied'] = False).
        Assumes the counter is level in the pelvis frame (scene.yaml); camera yaw / x / y errors stay."""
        cfg = self.cfg.get("plane_refine") or {}
        rep = {"applied": False}
        if not cfg.get("enabled", False):
            return np.eye(3), 0.0, rep
        sc = self.scene
        v, Xs, Ys, Zs = valid, P[..., 0], P[..., 1], P[..., 2]
        pm_x, pm_y = Ys - sc.origin[1], Xs - sc.origin[0]
        W, D = sc.cfg["counter"]["mat_size"]
        bins_near = min(cy for _, cy in sc.cfg["bins"]["centers"].values()) - sc.cfg["bins"]["outer_size"][1] / 2
        # stay clear of the bins (their front walls are vertical surfaces inside the search band) and mat edges
        sel = v & (np.abs(Zs - sc.counter_z) < cfg.get("search_dz", 0.03)) & (pm_x > 0.02) & (pm_x < W - 0.02) \
            & (pm_y > 0.0) & (pm_y < bins_near - cfg.get("bin_margin", 0.04))
        x, y, z = Xs[sel].astype(np.float64), Ys[sel].astype(np.float64), Zs[sel].astype(np.float64)
        rep["n"] = int(len(x))
        if len(x) < cfg.get("min_points", 1500):
            rep["why"] = "few_points"
            return np.eye(3), 0.0, rep
        # RANSAC (fixed seed: deterministic) for the inliers, then least squares on them. Plain trimmed least squares
        # can be pulled off by vertical surfaces (object sides, bin walls) before it gets to trim them.
        A = np.c_[x, y, np.ones(len(x))]
        rng = np.random.default_rng(0)
        thr = cfg.get("inlier_m", 0.003)
        best = None
        for _ in range(int(cfg.get("ransac_iters", 40))):
            idx = rng.choice(len(x), 3, replace=False)
            try:
                coef = np.linalg.solve(A[idx], z[idx])
            except np.linalg.LinAlgError:
                continue
            n_in = np.count_nonzero(np.abs(A @ coef - z) < thr)
            if best is None or n_in > best[0]:
                best = (n_in, coef)
        coef = best[1]
        # drop candidates next to anything clearly ABOVE this first plane (object sides, mixed edge pixels): such
        # points only ever lie above the counter, so inside a wide band they would bias the fit upwards
        import cv2
        above = (v & (Zs - (coef[0] * Xs + coef[1] * Ys + coef[2]) > cfg.get("high_m", 0.012))).astype(np.uint8)
        k = 2 * int(cfg.get("high_margin_cells", 2)) + 1
        clear = (cv2.dilate(above, np.ones((k, k), np.uint8)) == 0)[sel]
        if clear.sum() >= cfg.get("min_points", 1500):
            A, z = A[clear], z[clear]
        # least-squares refits with an ADAPTIVE band (2.5 x robust sigma of the residuals, at least inlier_m): a fixed
        # band as narrow as the depth noise cuts the residuals of a slightly tilted first plane asymmetrically and
        # keeps the tilt (biased fit)
        for _ in range(4):
            r = A @ coef - z
            near = np.abs(r) < 0.015
            sig = 1.4826 * np.median(np.abs(r[near])) if near.any() else thr
            keep = np.abs(r) < max(thr, 2.5 * sig)
            coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        a, b, c0 = coef
        n = np.array([-a, -b, 1.0])
        n /= np.linalg.norm(n)
        tilt = float(np.degrees(np.arccos(n[2])))
        # rotation taking n to +z (axis n x z), about the camera centre
        axis = np.cross(n, [0.0, 0.0, 1.0])
        sa = np.linalg.norm(axis)
        if sa < 1e-12:
            Rc = np.eye(3)
        else:
            k = axis / sa
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rc = np.eye(3) + sa * Kx + (1 - n[2]) * Kx @ Kx
        # plane n . P = kk; after rotation about o the plane is z = o_z + kk - n . o
        kk = c0 * n[2]
        z_plane = o[2] + kk - n @ o
        dz = sc.counter_z - z_plane
        rep.update({"tilt_deg": tilt, "dz": float(dz), "inliers": int(keep.sum())})
        if tilt > cfg.get("max_tilt_deg", 3.0) or abs(dz) > cfg.get("max_dz", 0.03):
            rep["why"] = "out_of_limits"
            return np.eye(3), 0.0, rep
        rep["applied"] = True
        return Rc, float(dz), rep

    def _bin_refinement(self, valid, Pn, R, o, R0, o0):
        """Camera yaw and x / y correction from the KNOWN bins (scene.yaml): depth points at bin-rim height near the
        bin footprints are matched (2D ICP, trimmed) to the inner and outer wall faces of the three boxes; the rigid
        2D transform (rotation about +z, translation) that aligns them is applied to the camera.

        Inputs: valid (h, w) and nominal-camera points Pn (h, w, 3) of a sparse pixel grid; current (plane-refined)
        camera (R, o) and nominal camera (R0, o0).
        Returns (R, o, report). Not applied (identity) when disabled, with too few rim points, a poor fit or a
        correction beyond the limits. Assumes the bins stand where scene.yaml says (fixed, measured layout)."""
        cfg = self.cfg.get("bin_refine") or {}
        rep = {"applied": False}
        if not cfg.get("enabled", False):
            return R, o, rep
        sc = self.scene
        b = sc.cfg["bins"]
        wh = float(b["wall_height"])
        band = float(cfg.get("height_band", 0.008))
        # nominal-camera points -> current (plane-refined) camera: P = o + (R R0^T) (Pn - o0)
        P = (o + (Pn[valid] - o0) @ (R @ R0.T).T)
        hz = P[:, 2] - sc.counter_z
        q = P[(hz > wh - band) & (hz < wh + band), :2]
        if len(q) < cfg.get("min_points", 300):
            rep.update(n=int(len(q)), why="few_points")
            return R, o, rep
        rects = []                                   # per bin: (centre pelvis xy, [outer, inner] half sizes)
        for cx, cy in b["centers"].values():
            c = sc.mat_to_pelvis([cx, cy, 0.0])[:2]
            hx, hy = b["outer_size"][1] / 2, b["outer_size"][0] / 2   # pelvis x = mat y
            rects.append((c, [np.array([hx - inset, hy - inset]) for inset in (0.0, float(b["wall_thickness"]))]))
        # each point belongs to the bin whose (outer) footprint + search margin contains it; others are dropped
        owner = np.full(len(q), -1)
        for k, (c, (h_out, _)) in enumerate(rects):
            owner[np.all(np.abs(q - c) <= h_out + cfg.get("search_m", 0.03), axis=1)] = k
        q, owner = q[owner >= 0], owner[owner >= 0]
        nmax = int(cfg.get("max_points", 500))
        if len(q) > nmax:                                # deterministic thinning (every k-th point)
            idx = np.linspace(0, len(q) - 1, nmax).astype(int)
            q, owner = q[idx], owner[idx]
        if len(q) < cfg.get("min_points", 300):
            rep.update(n=int(len(q)), why="few_points_near_bins")
            return R, o, rep
        # point-to-line ICP: minimise the distance of each point to its nearest wall face ALONG that face's normal,
        # linearised in the yaw angle th: R(th) q ~ q + th * (-q_y, q_x). Converges in a few iterations and lets
        # points slide along the walls (a point-to-point fit drags them and under-corrects).
        th, t = 0.0, np.zeros(2)
        keep = np.ones(len(q), bool)
        # coarse-to-fine outlier gate (m): a tight gate from the start would discard exactly the points that carry
        # the correction (e.g. the side walls under a lateral camera shift)
        gates = [0.03, 0.02, 0.012, 0.008, 0.006, 0.004]
        for it in range(int(cfg.get("iterations", 8))):
            c_, s_ = np.cos(th), np.sin(th)
            qq = q @ np.array([[c_, s_], [-s_, c_]]) + t
            m, nrm = _nearest_on_bins(qq, owner, rects)
            r = np.einsum("ij,ij->i", qq - m, nrm)               # signed distance along the face normal
            keep = np.abs(r) < max(3.0 * np.median(np.abs(r)), gates[min(it, len(gates) - 1)])
            A = np.c_[np.einsum("ij,ij->i", nrm, np.c_[-qq[:, 1], qq[:, 0]]), nrm][keep]
            sol, *_ = np.linalg.lstsq(A, -r[keep], rcond=None)
            dth, dt = sol[0], sol[1:]
            # compose: new transform = small step (rotation dth about the origin, then dt) after the current one
            th = th + dth
            Rd = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
            t = Rd @ t + dt
        c_, s_ = np.cos(th), np.sin(th)
        qq = q @ np.array([[c_, s_], [-s_, c_]]) + t
        m, nrm = _nearest_on_bins(qq, owner, rects)
        rms = float(np.sqrt(np.mean(np.einsum("ij,ij->i", qq - m, nrm)[keep] ** 2)))
        rep.update(n=int(len(q)), inliers=int(keep.sum()), yaw_deg=float(np.degrees(th)), dxy=[float(t[0]), float(t[1])],
                   rms_mm=rms * 1000)
        if (keep.sum() < cfg.get("min_points", 300) or rms > cfg.get("max_rms", 0.004)
                or abs(np.degrees(th)) > cfg.get("max_yaw_deg", 3.0) or np.linalg.norm(t) > cfg.get("max_shift", 0.03)):
            rep["why"] = "fit_rejected"
            return R, o, rep
        Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        rep["applied"] = True
        return Rz @ R, Rz @ o + np.array([t[0], t[1], 0.0]), rep

    def _fitters(self):
        if self._top_fitter is None:
            self._top_fitter = TopFaceFitter(self.scene, **(self.shape_cfg.get("polygon_grid") or {}))
            self._sil_fitter = SilhouetteFitter(self.scene, self.cal, self.cam, height=self.height)
        return self._top_fitter, self._sil_fitter

    def _detect_v2(self, rgb, depth, debug=False):
        import cv2
        c = self.cfg
        self._full_shape = rgb.shape[:2]
        win = self._object_roi(rgb.shape[:2])
        v0, v1, u0, u1 = win
        self._win_off = np.array([u0, v0], float)       # ROI-local pixel + this = full-image pixel
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[v0:v1, u0:u1]), cv2.COLOR_RGB2HSV)
        geo = self._depth_geometry(depth, win)
        self._geo = geo
        self.last_plane = geo.plane if geo is not None else None
        hull_keep = self._bin_hulls_mask(rgb.shape[:2])[v0:v1, u0:u1]
        top_fit, sil_fit = self._fitters()
        if geo is not None:                     # render-and-compare uses the plane-refined camera
            sil_fit.set_camera(geo.R, geo.o)
        else:
            T = self.cal._T(self.cam)
            sil_fit.set_camera(T[:3, :3], T[:3, 3])
        dets, rejected = [], []
        k = int(c["open_px"])
        kernel = np.ones((k, k), np.uint8) if k > 0 else None
        strict = {col: self._hsv_mask(hsv, r, open_px=0) for col, r in c["hsv"].items()}
        relaxed = {col: self._hsv_mask(hsv, c.get("hsv_relaxed", c["hsv"])[col], open_px=0) for col in c["hsv"]}
        regions = []                                    # (colour, blob mask in roi, roi slices, silhouette source)
        if geo is None:
            # no depth stream: per-colour blobs from the STRICT ranges (colour is the only evidence); the silhouette
            # for render-and-compare may use the relaxed ranges (it only grows from a strict blob)
            for col in c["hsv"]:
                m = strict[col] & hull_keep
                if kernel is not None:
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
                sil = (relaxed[col] | strict[col]) & hull_keep
                for blob, roi in self._components(m):
                    regions.append((col, blob, roi, sil))
        else:
            # geometry first: object-height regions (+ strict-colour pixels where depth is missing), then colour
            hole = (~geo.valid).astype(np.uint8) * 255 & hull_keep
            any_strict = np.zeros_like(hull_keep)
            labels = np.zeros(hull_keep.shape, np.uint8)  # 1.. = colour index (relaxed ranges, disjoint hues)
            cols = list(c["hsv"])
            for ci, col in enumerate(cols):
                any_strict |= strict[col]
                labels[relaxed[col] > 0] = ci + 1
            wb_on = (c.get("colour") or {}).get("mode", "hsv") == "wb_proto"
            if wb_on:
                lab_wb, sat_wb = self._wb_labels(rgb, win, cols)
                cast = float(np.abs(np.log(self.last_wb)).max())
                wb_on = cast >= np.log((c.get("colour") or {}).get("wb_min_cast", 1.0))
                if wb_on:                   # a real colour cast: white-balanced labels; else keep the HSV ones
                    labels = lab_wb
            # (the projected bin hulls are NOT applied to depth pixels: an object in front of a bin overlaps the
            # bin's image hull; bins are excluded in 3D. They only guard pixels without depth: `hole`)
            cand = geo.obj | (any_strict & hole)
            if kernel is not None:
                cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)
            sat = sat_wb if wb_on else hsv[..., 1]
            hmar = self._hue_margin(self._wb_hsv if wb_on else hsv, cols)
            for comp, roi in self._components(cand):
                for col, blob, why in self._colour_regions(comp, roi, labels[roi], sat[roi], cols, geo):
                    if why:
                        rejected.append((None, why))
                        continue
                    if col is None:                 # uncoloured object-height region (keep_uncoloured)
                        sil = geo.not_counter & (geo.valid | (hull_keep > 0))
                    else:
                        sil = (relaxed[col] | strict[col]) & geo.not_counter & (geo.valid | (hull_keep > 0))
                    regions.append((col, blob, roi, sil))
        seg = c.get("segmenter", "geometry")
        hints = {}
        if seg != "geometry":                   # A3 learned candidates: regions from a model, depth fit as v2
            regions, hints = self._external_regions(seg, rgb, win, geo, hsv, hull_keep)
        for col, blob, roi, sil in regions:
            area = int(np.count_nonzero(blob))
            vv, uu = np.nonzero(blob)
            cent = np.array([uu.mean() + roi[1].start, vv.mean() + roi[0].start]) + self._win_off
            if id(blob) in hints:
                cconf = hints[id(blob)][1]
            elif geo is None:
                cconf = 1.0
            elif col is None:
                cconf = 0.0
            else:
                cconf = self._colour_share(col, blob, roi, labels, sat, cols)
                if (c.get("colour") or {}).get("conf", "share") == "share_x_margin":
                    sel = blob & (labels[roi] == cols.index(col) + 1)
                    cconf *= float(np.median(hmar[roi][sel])) if sel.any() else 0.0
            for det in self._blob_v2(col, blob, roi, cent, area, geo, sil):
                det.color_conf = cconf
                if id(blob) in hints:
                    det._hue = hints[id(blob)][2]
                if id(blob) in hints and det.shape is None and hints[id(blob)][1] >= c.get("seg_shape_min", 0.5):
                    det.shape, det.method = hints[id(blob)][0], det.method + "+model"
                why = self._reject(det)
                if why is None and col is None:     # uncoloured: only a clear prism fit makes it one of ours
                    best = max(det.fit_iou.values()) if det.fit_iou else 0.0
                    if det.shape is None or best < (c.get("colour") or {}).get("uncoloured_min_iou", 0.8):
                        why = f"uncoloured_no_fit {best:.2f}"
                if why is None and det.fit_iou:
                    if max(det.fit_iou.values()) < self.shape_cfg.get("reject_below_iou", 0.0):
                        why = "no_shape_fit"
                if why is None and geo is not None and det.method == "plane":
                    # the frame HAS depth but this blob has no top face at object height (colour fringe in a depth
                    # hole / stereo band, a fragment): only a clear silhouette fit makes it an object
                    best = max(det.fit_iou.values()) if det.fit_iou else 0.0
                    if best < self.shape_cfg.get("min_fit_without_top", 0.5):
                        why = f"no_top_no_fit {best:.2f}"
                if why:
                    rejected.append((det, why))
                else:
                    dets.append(det)
        dets, frag = self._drop_fragments(dets)
        rejected += frag
        if seg != "geometry":                   # overlapping model regions on one object: keep the best one
            dets.sort(key=lambda d: -d.color_conf)
            kept = []
            for d in dets:
                if all(np.hypot(*(d.top[:2] - k.top[:2])) >= c.get("seg_merge_m", 0.025) for k in kept):
                    kept.append(d)
                else:
                    rejected.append((d, "duplicate_region"))
            dets = kept
        if seg != "geometry" and c.get("colour_from") == "hue_cluster":
            self._hue_colours(dets)
        if c.get("vlm_verify"):
            self._vlm_verify(rgb, dets)
        min_cc = float((c.get("colour") or {}).get("min_conf", 0.0))
        for d in dets:                          # abstain: an unsure colour is None (costs 0), never a guess (-1)
            if d.color_conf < min_cc:
                d.color = None
        return (dets, rejected) if debug else dets

    def _wb_labels(self, rgb, win, cols):
        """A3 candidate `v2-colour-invariant` (perception.colour.mode: wb_proto).
        1. Illuminant from the KNOWN bins: inside the bins' image hulls, the more saturated half of the pixels
           (bin walls and floors; the grey counter in the hull margin is dropped) has median RGB m; the bins'
           albedo is scene bins.rgb, so the per-channel gains g = albedo / m map the frame back to the reference
           light (von Kries). Gains are normalised to green = 1 and clipped to [1/gmax, gmax].
        2. Per object pixel: hue of the corrected RGB; label = nearest object-colour hue (scene objects.colors in
           the same space), only for pixels with corrected saturation >= s_min and value >= v_min. No fixed hue
           thresholds, so a global colour cast moves pixels and prototypes together.
        Returns (labels uint8 (window) 0 = none / 1.. = colour index, saturation uint8 of the corrected image)."""
        import cv2
        cc = self.cfg.get("colour") or {}
        v0, v1, u0, u1 = win
        g = self._wb_gains(rgb, cc)
        self.last_wb = g
        w = np.clip(rgb[v0:v1, u0:u1].astype(np.float32) * g[None, None, :], 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(w, cv2.COLOR_RGB2HSV)
        self._wb_hsv = hsv
        h = hsv[..., 0].astype(np.float32)
        protos = []
        for col in cols:
            ref = np.uint8([[self.scene.colors[col]]])
            protos.append(float(cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)[0, 0, 0]))
        dist = np.stack([np.minimum(np.abs(h - p), 180 - np.abs(h - p)) for p in protos])
        lab = (np.argmin(dist, 0) + 1).astype(np.uint8)
        ok = (hsv[..., 1] >= cc.get("wb_s_min", 50)) & (hsv[..., 2] >= cc.get("wb_v_min", 30))
        ok &= dist.min(0) <= cc.get("wb_max_hue_dist", 25)
        return np.where(ok, lab, 0).astype(np.uint8), hsv[..., 1]

    def _external_regions(self, seg, rgb, win, geo, hsv, hull_keep):
        """Regions [(colour, blob, roi, sil)] + {id(blob): (shape hint, score)} from a learned segmenter."""
        import cv2
        from aaltonomous.baselines.classical.loop.vision import learned
        if seg == "detector":
            regs = learned.detector_regions(rgb, self.cfg["detector_weights"], self.cfg.get("detector_conf", 0.35))
        elif seg == "open_vocab":
            regs = learned.open_vocab_regions(rgb, self.cfg.get("open_vocab_thr", 0.15),
                                              words=self.cfg.get("open_vocab_words"))
        else:
            raise ValueError(f"perception.segmenter {seg!r}")
        v0, v1, u0, u1 = win
        out, hints = [], {}
        for col, shp, score, m in regs:
            mw = m[v0:v1, u0:u1]
            if geo is not None:                 # with depth: only the model's pixels at object height count
                b = mw & (geo.obj > 0)          # (a flat sticker / texture the model fires on has none)
            else:                               # no depth: the model's saturated pixels outside the bins
                b = mw & (hsv[..., 1] >= 60) & (hull_keep > 0)
            if np.count_nonzero(b) < self.cfg["min_area_px"]:
                continue
            x, y, w, h = cv2.boundingRect(b.astype(np.uint8))
            roi = (slice(y, y + h), slice(x, x + w))
            blob = b[roi].copy()
            sil = (b.astype(np.uint8) * 255)
            out.append((col, blob, roi, sil))
            hue = None
            if self.cfg.get("colour_from") == "hue_cluster":   # hue of the model's own region (not depth-cut)
                src = rgb
                if (self.cfg.get("hue") or {}).get("white_balance"):   # undo a cast first (gains from the bins)
                    g = self._wb_gains(rgb, self.cfg.get("colour") or {})
                    if float(np.abs(np.log(g)).max()) >= np.log((self.cfg.get("colour") or {}).get("wb_min_cast", 1.3)):
                        src = np.clip(rgb.astype(np.float32) * g[None, None, :], 0, 255).astype(np.uint8)
                full = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
                hue = learned.region_hue(full, m & (self._bin_hulls_mask(rgb.shape[:2]) > 0) if
                                         self.cfg.get("hue_skip_bins", True) else m)
            hints[id(blob)] = (shp, score, hue)
        return out, hints

    def _hue_colours(self, dets):
        """Detector + hue hybrid: colour from the regions' own hues, assigned jointly per frame
        (learned.assign_colours) against the selected hue prototypes. Replaces the model's colour label."""
        import cv2
        from aaltonomous.baselines.classical.loop.vision import learned
        hc = self.cfg.get("hue") or {}
        sets = hc.get("prototype_sets") or {}
        protos = sets.get(hc.get("prototypes", "scene"))
        if not protos:                          # "scene": the object colours of scene.yaml
            protos = {k: float(cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
                      for k, v in self.scene.colors.items()}
        res = learned.assign_colours([getattr(d, "_hue", None) for d in dets], protos, hc.get("gap", 10.0),
                                     hc.get("max_dist", 25.0))
        agree = hc.get("agree_with_model", False)
        for d, (col, dist) in zip(dets, res):
            if agree and col is not None and col != d.color:
                col = None                      # hue and model disagree: abstain (0), never pick one (-1 risk)
            d.color = col
            d.color_conf = 0.0 if col is None else float(1 - dist / max(hc.get("max_dist", 25.0), 1e-6))

    def _vlm_verify(self, rgb, dets):
        """A3 `vlm-verify`: second opinion on unsure detections only (see learned.py)."""
        from aaltonomous.baselines.classical.loop.vision import learned
        below = float(self.cfg.get("vlm_below_conf", 0.9))
        for d in dets:
            if d.shape is not None and d.color_conf >= below:
                continue
            col, shp = learned.vlm_name(learned.crop(rgb, d.uv))
            d.method += "+vlm"
            if d.shape is None and shp is not None:
                d.shape = shp
            if d.color is not None and col is not None and col != d.color:
                d.color = None                  # disagreement: abstain (0), never swap to the VLM's colour

    def _hue_margin(self, hsv, cols):
        """Per pixel: (d2 - d1) / d2 of the hue distances to the nearest / second-nearest object-colour hue
        (scene objects.colors), 0..1. Near 0 = the hue sits between two object colours (e.g. orange between red
        and yellow): a colour read there is a coin toss."""
        import cv2
        h = hsv[..., 0].astype(np.float32)
        protos = [float(cv2.cvtColor(np.uint8([[self.scene.colors[c]]]), cv2.COLOR_RGB2HSV)[0, 0, 0]) for c in cols]
        d = np.sort(np.stack([np.minimum(np.abs(h - p), 180 - np.abs(h - p)) for p in protos]), 0)
        return (d[1] - d[0]) / np.maximum(d[1], 1e-3)

    def _wb_gains(self, rgb, cc):
        import cv2
        keep = self._bin_hulls_mask(rgb.shape[:2])
        px = rgb[::4, ::4][keep[::4, ::4] == 0].astype(np.float32)
        gmax = float(cc.get("wb_gain_max", 3.0))
        if len(px) < 200:
            return np.ones(3, np.float32)
        hsv = cv2.cvtColor(px.astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0]
        sel = px[(hsv[:, 1] >= np.median(hsv[:, 1])) & (hsv[:, 2] >= 20)]
        if len(sel) < 100:
            return np.ones(3, np.float32)
        m = np.maximum(np.median(sel, 0), 1.0)
        ref = np.asarray(self.scene.bin_rgb, np.float32)
        g = ref / m
        g = g / g[1]
        return np.clip(g, 1 / gmax, gmax).astype(np.float32)

    @staticmethod
    def _colour_share(col, blob, roi, labels, sat, cols) -> float:
        """Share of colour `col` in the saturation-weighted relaxed-label vote over `blob` (window `roi`)."""
        lab = labels[roi][blob]
        w = np.bincount(lab.ravel(), weights=sat[roi][blob].ravel().astype(float), minlength=len(cols) + 1)[1:]
        tot = w.sum()
        return float(w[cols.index(col)] / tot) if tot > 0 else 0.0

    def _components(self, m):
        """Connected components (8-connected) of a uint8 mask with at least min_area_px pixels.
        Yields (blob bool mask of its bounding box, (row slice, col slice)) in the mask's coordinates."""
        import cv2
        bx, by, bw, bh = cv2.boundingRect(m)            # label only the part of the image with pixels
        if bw == 0 or bh == 0:
            return
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m[by:by + bh, bx:bx + bw], connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.cfg["min_area_px"]:
                continue
            x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                      cv2.CC_STAT_HEIGHT))
            yield lab[y:y + h, x:x + w] == i, (slice(by + y, by + y + h), slice(bx + x, bx + x + w))

    def _colour_regions(self, comp, roi, labels, sat, cols, geo):
        """Split one object-height region into objects and give each its colour.

        Colour = saturation-weighted majority vote of the region's coloured pixels (relaxed ranges): washed-out
        (white), very dark or colour-shifted top faces carry little or wrong evidence, the saturated side faces
        decide. If only one colour is substantial (>= colour.split_frac of the coloured weight), the region is one
        object. If several are, the region holds several touching objects: they are separated in 3D (groups of
        top-face points on the counter, TopFaceFitter.split; every region pixel goes to the nearest group) and each
        group gets its own vote. A region with too little colour is not one of our objects (robot hand, occluder).
        Returns [(colour, blob mask (roi), reject reason or None)]."""
        cc = self.cfg.get("colour") or {}
        area = int(np.count_nonzero(comp))
        lab = np.where(comp, labels, 0)
        coloured = int(np.count_nonzero(lab))
        if coloured < max(cc.get("min_coloured_px", 40), cc.get("min_coloured_frac", 0.1) * area):
            if cc.get("keep_uncoloured"):       # A3 `cloud`: geometry decides objecthood, colour only labels it
                return [(None, comp, None)]
            return [(None, None, f"uncoloured {coloured}/{area}px")]

        def vote(mask):
            w = np.bincount(lab[mask].ravel(), weights=sat[mask].ravel().astype(float), minlength=len(cols) + 1)[1:]
            return w

        w = vote(comp)
        sig = [i for i in range(len(cols)) if w[i] >= cc.get("split_frac", 0.25) * w.sum()]
        if len(sig) <= 1:
            return [(cols[int(np.argmax(w))], comp, None)]
        sel = comp & (geo.obj[roi] > 0)
        if np.count_nonzero(sel) < self.cfg["min_valid_depth_px"]:
            return [(cols[int(np.argmax(w))], comp, None)]
        P = self._corrected_points(geo, roi, sel)
        top = self._top_face(P[:, 2])
        if top.sum() < self.shape_cfg.get("min_top_px", 20):
            return [(cols[int(np.argmax(w))], comp, None)]
        groups = self._top_fitter.split(P[top, :2], self.shape_cfg.get("min_top_px", 20))
        if len(groups) <= 1:
            return [(cols[int(np.argmax(w))], comp, None)]
        centres = np.array([P[top][g, :2].mean(0) for g in groups])
        nearest = np.argmin(np.linalg.norm(P[:, None, :2] - centres[None], axis=2), axis=1)
        vv, uu = np.nonzero(sel)
        out = []
        for k in range(len(groups)):
            blob = np.zeros_like(comp)
            blob[vv[nearest == k], uu[nearest == k]] = True
            if np.count_nonzero(blob) < self.cfg["min_area_px"]:
                continue
            wk = vote(blob)
            if wk.sum() <= 0:
                continue
            out.append((cols[int(np.argmax(wk))], blob, None))
        return out or [(cols[int(np.argmax(w))], comp, None)]

    def _drop_fragments(self, dets):
        """A blob with no shape fit (best IoU below `fragment_iou`) next to a well-fitted object of the same colour
        (closer than `fragment_radius`) is a piece of that object (e.g. a darker side face cut off from its top
        face by the colour threshold), not an object. Isolated poor fits are kept: they can be occluded objects."""
        thr = self.shape_cfg.get("fragment_iou", 0.3)
        rad = self.shape_cfg.get("fragment_radius", 0.045)
        keep, frag = [], []
        good = [d for d in dets if d.fit_iou and max(d.fit_iou.values()) >= thr]
        for d in dets:
            poor = not d.fit_iou or max(d.fit_iou.values()) < thr
            near = any(g.color == d.color and np.hypot(*(g.top[:2] - d.top[:2])) < rad for g in good if g is not d)
            (frag if poor and near else keep).append((d, "fragment") if poor and near else d)
        return keep, frag

    def _blob_v2(self, color, blob, roi, cent, area, geo, sil_src) -> list:
        """One colour blob (bool mask of the window `roi`, centroid `cent` in full-image pixels) -> Detections
        (position, shape, yaw) with the configured shape method. Usually one; more when the blob's top-face points
        form separate groups on the counter (two objects of the same colour that touch in the image)."""
        c = self.cfg
        y0, x0 = roi[0].start, roi[1].start
        vv, uu = np.nonzero(blob)
        uv = (float(np.median(uu)) + x0 + self._win_off[0], float(np.median(vv)) + y0 + self._win_off[1])
        if geo is not None:
            sel = blob & (geo.obj[roi] > 0)
            n_depth = int(sel.sum())
            if n_depth >= c["min_valid_depth_px"]:
                Zb = geo.Z[roi][sel]
                tsel = self._top_face(Zb)
                # full refined 3D for the top-face points only
                P = self._corrected_points(geo, roi, sel)[tsel]
                if len(P) >= self.shape_cfg.get("min_top_px", 20):
                    w8 = geo.d[roi][sel][tsel] ** 2          # pixel footprint on the top face ~ depth^2
                    groups = [np.arange(len(P))]
                    mtp = self.shape_cfg.get("min_top_px", 20)
                    if self.shape_cfg.get("split_blobs", True):
                        groups = self._top_fitter.split(P[:, :2], mtp)
                        ext = self.shape_cfg.get("split_extent")
                        if len(groups) == 1 and ext:         # touching tops of the same height: cut at the neck
                            groups = self._top_fitter.split_neck(P[:, :2], float(ext), mtp)
                    if len(groups) > 1:                      # the silhouette holds several objects
                        restrict = self.shape_cfg.get("split_restrict", False)
                        out = []
                        for k, g in enumerate(groups):
                            own = self._own_cells(P[:, :2], groups, k) if restrict else None
                            out.append(self._from_top_points(color, P[g], w8[g], int(area * len(g) / len(P)),
                                                             n_depth, uv, blob if restrict else None,
                                                             roi if restrict else None, None, own=own))
                        return out
                    return [self._from_top_points(color, P, w8, area, n_depth, uv, blob, roi, sil_src)]
        # no usable depth on this blob: centroid ray on the known counter plane + render-and-compare
        xy0 = self._ray_plane(cent)
        if xy0 is None:
            return [Detection(color, np.full(3, np.nan), area, 0, "plane", None, uv)]
        rend = self._render_fit(blob, roi, sil_src, xy0)
        shape, yaw, conf, ious, xy = self._combine("render", None, rend)
        xy = xy if xy is not None else xy0
        top = np.array([xy[0] + self.xy_offset[0], xy[1] + self.xy_offset[1], self.scene.counter_z + self.height])
        return [Detection(color, top, area, int(geo is not None and (blob & (geo.obj[roi] > 0)).sum()), "plane", yaw,
                          uv, shape, conf, ious)]

    @staticmethod
    def _own_cells(xy_all, groups, k, far_m=0.006):
        """cells (M,2) -> bool: the cell belongs to group k (its nearest top point is in group k) or is far from
        every top point (free counter: keeps its 'empty' evidence)."""
        from scipy.spatial import cKDTree
        tree = cKDTree(xy_all)
        lab = np.empty(len(xy_all), int)
        for j, g in enumerate(groups):
            lab[g] = j

        def own(cells):
            d, i = tree.query(cells)
            return (lab[i] == k) | (d > far_m)
        return own

    def _from_top_points(self, color, P, w8, area, n_depth, uv, blob, roi, sil_src, own=None):
        """Detection from the refined 3D top-face points P (N,3) of one object (weights w8 ~ pixel footprint).
        own: optional cells -> bool; cells outside it (another object's top in the same blob) are unknown."""
        method = self.shape_cfg.get("method", "combined")
        z = float(np.median(P[:, 2]))
        xy = (P[:, :2] * w8[:, None]).sum(0) / w8.sum()
        poly = rend = None
        if method in ("polygon", "combined"):
            pc = self.shape_cfg.get("polygon") or {}
            observe = self._observer(z, blob, roi) if pc.get("visibility", True) else None
            if observe is not None and own is not None:
                base = observe

                def observe(cells, base=base):
                    occ, known = base(cells)
                    m = own(cells)
                    return occ & m, known & m
            args = (pc.get("min_iou", 0.55), pc.get("min_margin", 0.04))
            hid = (pc.get("min_known", 0.6), pc.get("hidden_vis", 1.0), pc.get("hidden_margin"), pc.get("hidden_min_iou"))
            poly = self._top_fitter.fit(P[:, :2], *args, pc.get("shift_mm", 3), observe, *hid)
            rf = pc.get("refit_hidden")
            if rf and observe is not None and poly.vis < rf.get("below_vis", 0.97):
                # partly hidden: the visible top points' centroid is off the object centre by up to ~half a
                # size, beyond the +-shift_mm search -> refit on a larger grid with a wide xy search
                if self._wide_fitter is None:
                    self._wide_fitter = TopFaceFitter(self.scene, half_m=rf.get("half_m", 0.056),
                                                      **{k: v for k, v in (self.shape_cfg.get("polygon_grid") or {}).items()
                                                         if k != "half_m"})
                poly = self._wide_fitter.fit(P[:, :2], *args, rf.get("shift_mm", 14), observe, *hid)
        if method in ("render", "combined") and blob is not None and own is None:
            rend = self._render_fit(blob, roi, sil_src, xy)
        if poly is None and rend is None:
            shape, yaw, conf, ious, fxy = None, None, 0.0, None, None
        else:
            shape, yaw, conf, ious, fxy = self._combine(method if rend is not None else "polygon", poly, rend)
        if fxy is not None and self.shape_cfg.get("position_from_fit", False):
            xy = fxy
        top = np.array([xy[0] + self.xy_offset[0], xy[1] + self.xy_offset[1], z])
        return Detection(color, top, area, n_depth, "depth", yaw, uv, shape, conf, ious,
                         visible=poly.vis if poly is not None else None)

    def _observer(self, z_top, blob, roi):
        """Callback for TopFaceFitter.fit: what does the camera see at grid cells (M,2) on the plane z = z_top?
        Each cell is projected with the refined camera; at that pixel the measured depth is compared with the
        cell's depth (tolerance top_face.tol_m or tol_rel x depth):
          occupied  measured = cell depth (the top surface is there) and the pixel belongs to this blob
          unknown   no valid depth, outside the image window, or measured NEARER than the cell (hidden behind
                    an occluder / a taller thing)
          empty     otherwise (the camera sees past the cell, e.g. the counter)
        Returns (occupied, known)."""
        geo = self._geo
        tcfg = self.shape_cfg.get("visibility") or {}
        tol_m, tol_rel = tcfg.get("tol_m", 0.004), tcfg.get("tol_rel", 0.01)
        K = self.cal.cams[self.cam].K
        v_off, u_off = int(self._win_off[1]), int(self._win_off[0])
        H, W = geo.valid.shape

        def observe(cells):
            P = np.c_[cells, np.full(len(cells), z_top)]
            p = (P - geo.o) @ geo.R                                  # optical frame of the refined camera
            u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int) - u_off
            v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int) - v_off
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (p[:, 2] > 0)
            uu, vv = np.where(inside, u, 0), np.where(inside, v, 0)
            ok = inside & geo.valid[vv, uu]
            meas = geo.d[vv, uu]
            tol = np.maximum(tol_m, tol_rel * p[:, 2])
            on_top = ok & (np.abs(meas - p[:, 2]) <= tol)
            if blob is not None:
                bv, bu = vv - roi[0].start, uu - roi[1].start
                inb = (bv >= 0) & (bv < blob.shape[0]) & (bu >= 0) & (bu < blob.shape[1])
                on_top &= inb & blob[np.where(inb, bv, 0), np.where(inb, bu, 0)]
            hidden = ok & (meas < p[:, 2] - tol)
            return on_top, ok & ~hidden
        return observe

    def _top_face(self, z):
        """Which of the heights z (pelvis, m; refined camera) belong to an object's top face. The top is searched
        near the KNOWN top height z_ref = counter + object height (top_search below, top_tolerance above):
        anything higher is something else in the same image region (an occluder, a hand above the object). Within
        that window the robust top (top_percentile) is taken and points within top_band below it (and
        top_tolerance above it) are the top face."""
        c = self.cfg
        z_ref = self.scene.counter_z + self.height
        tol = c.get("top_tolerance", 0.004)
        win = (z >= z_ref - c.get("top_search", 0.015)) & (z <= z_ref + tol)
        if not win.any():
            return np.zeros(len(z), bool)
        z_top = np.percentile(z[win], c["top_percentile"])
        return (z >= z_top - c["top_band"]) & (z <= min(z_top + tol, z_ref + tol))

    def _corrected_points(self, geo, roi, sel):
        """(N,3) pelvis points of the selected ROI pixels with the plane-refined camera."""
        v0, v1, u0, u1 = geo.win
        rays = self._pixel_rays(self._full_shape)[:, v0:v1, u0:u1]
        d = geo.d[roi][sel].astype(np.float64)
        ray = np.stack([rays[k][roi][sel] for k in range(3)], -1).astype(np.float64)   # nominal frame
        T = self.cal._T(self.cam)
        Rc = geo.R @ T[:3, :3].T                                                        # correction rotation
        return geo.o + (d[:, None] * ray) @ Rc.T

    def _ray_plane(self, uv):
        """Blob centroid ray of the (corrected) camera intersected with a plane 3/4 up the object (the centroid of
        a blob lies between the top face and the front face). None if the ray does not go down."""
        sil = self._sil_fitter
        K = self.cal.cams[self.cam].K
        ray = sil.R @ np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        z_aim = self.scene.counter_z + 0.75 * self.height
        if ray[2] >= -1e-6:
            return None
        s = (z_aim - sil.o[2]) / ray[2]
        return (sil.o + s * ray)[:2]

    def _render_fit(self, blob, roi, sil_src, xy0):
        """Render-and-compare on the colour silhouette around the blob (the blob itself lacks the low part of the
        side faces, which the object-height mask removes; the silhouette source keeps them)."""
        import cv2
        pad = 12
        H, W = sil_src.shape
        v0, v1 = max(roi[0].start - pad, 0), min(roi[0].stop + pad, H)
        u0, u1 = max(roi[1].start - pad, 0), min(roi[1].stop + pad, W)
        S = (sil_src[v0:v1, u0:u1] > 0).astype(np.uint8)
        seed = np.zeros_like(S)
        seed[roi[0].start - v0:roi[0].stop - v0, roi[1].start - u0:roi[1].stop - u0] = blob
        n, lab = cv2.connectedComponents(S | seed, connectivity=8)
        keep = np.unique(lab[seed > 0])
        B = np.isin(lab, keep[keep > 0]).astype(np.uint8)
        rc = self.shape_cfg.get("render") or {}
        return self._sil_fitter.fit(B, np.array([u0, v0], float) + self._win_off, xy0, rc.get("min_iou", 0.6),
                                    rc.get("min_margin", 0.03))

    def _combine(self, method, poly, rend):
        """Merge the fitter results into (shape, yaw, confidence, ious, fitted xy)."""
        if method == "polygon" and poly is not None:
            return poly.shape, poly.yaw, poly.margin, poly.iou, poly.xys.get(poly.shape, poly.xy)
        if poly is None and rend is not None:                       # render only (method render, or no depth)
            return rend.shape, rend.yaw, rend.margin, rend.iou, rend.xys.get(rend.shape, rend.xy)
        if rend is None:
            return None, None, 0.0, None, None
        # combined: mean IoU per shape, decision on the mean; yaw from the polygon fit when it agrees
        cc = self.shape_cfg.get("combined") or {}
        ious = {s: 0.5 * (poly.iou[s] + rend.iou[s]) for s in poly.iou if s in rend.iou}
        shape, margin = decide(ious, cc.get("min_iou", 0.55), cc.get("min_margin", 0.03))
        if shape is None:
            return None, None, margin, ious, poly.xy
        return shape, poly.yaws.get(shape), margin, ious, poly.xys.get(shape, poly.xy)

    # ================================================================================================ legacy pipeline
    def _detect_legacy(self, rgb, depth, debug=False):
        import cv2
        c = self.cfg
        dets, rejected = [], []
        dlo, dhi = c["depth_range_mm"]
        keep = self._keep_mask(depth)
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < c["min_area_px"]:
                    continue
                blob = (lab == i).astype(np.uint8)
                e = int(c["erode_px"])
                core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                if core.sum() < 0.25 * area:
                    core = blob
                v, u = np.nonzero(core)
                d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
                ok = (d >= dlo) & (d <= dhi)
                det = None
                if ok.sum() >= c["min_valid_depth_px"]:
                    det = self._from_depth(color, u[ok], v[ok], d[ok], area)
                if det is None:
                    det = self._from_plane(color, cent[i], area, int(ok.sum()))
                why = self._reject(det)
                if why:
                    rejected.append((det, why))
                else:
                    dets.append(det)
        return (dets, rejected) if debug else dets

    def _keep_mask(self, depth):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        keep = self._bin_hulls_mask(depth.shape if depth is not None else (480, 640)).copy()
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        ok = (h >= 0.25) & (h <= c["height_range"][1])
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _from_depth(self, color, u, v, d, area):
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        xy = xy + self.xy_offset
        return Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                         (float(np.median(u)), float(np.median(v))))

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.cal._T(self.cam)
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _nearest_on_bins(q, owner, rects):
    """For points q (N,2) assigned to bins (owner (N,) index into rects = [(centre, [outer half, inner half])]):
    the nearest point on the outer or inner wall face of the point's own bin and that face's unit normal.
    Vectorised over all points (per-point centre and half sizes)."""
    C = np.array([r[0] for r in rects])[owner]
    best_m, best_n, best_d = None, None, None
    for k in range(2):                                   # outer face, inner face
        Hh = np.array([r[1][k] for r in rects])[owner]
        m, n = _nearest_on_rect(q - C, Hh)
        dd = np.linalg.norm(q - C - m, axis=1)
        if best_d is None:
            best_m, best_n, best_d = m, n, dd
        else:
            upd = dd < best_d
            best_m[upd], best_n[upd], best_d[upd] = m[upd], n[upd], dd[upd]
    return best_m + C, best_n


def _nearest_on_rect(r, h):
    """Nearest point on the boundary of axis-aligned rectangles |x| <= h[..., 0], |y| <= h[..., 1] to points r (N,2)
    (relative to the rectangle centre; h (2,) or (N,2)), and the outward unit normal of that boundary point."""
    h = np.broadcast_to(h, r.shape)
    a = np.abs(r)
    m = np.clip(r, -h, h)                            # outside: clamping gives the nearest boundary point
    n = np.zeros_like(r)
    out_x, out_y = a[:, 0] > h[:, 0], a[:, 1] > h[:, 1]
    n[out_x & ~out_y, 0] = 1.0
    n[out_y & ~out_x, 1] = 1.0
    corner = out_x & out_y
    if corner.any():
        n[corner] = (a[corner] - h[corner]) / np.linalg.norm(a[corner] - h[corner], axis=1, keepdims=True)
    inside = ~out_x & ~out_y
    dx, dy = h[:, 0] - a[:, 0], h[:, 1] - a[:, 1]    # inside: snap to the nearer edge
    snap_x = inside & (dx <= dy)
    snap_y = inside & (dx > dy)
    m[snap_x, 0] = np.sign(r[snap_x, 0]) * h[snap_x, 0]
    m[snap_y, 1] = np.sign(r[snap_y, 1]) * h[snap_y, 1]
    n[snap_x, 0] = 1.0
    n[snap_y, 1] = 1.0
    return m, n * np.where(r < 0, -1.0, 1.0)        # normals point away from the rectangle centre
