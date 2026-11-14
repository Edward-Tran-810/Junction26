"""Pick-and-place planning for one detected object: Cartesian waypoints -> IK -> 30 Hz joint trajectories.

Sequence (one arm; the other holds its current target):
  home -> pregrasp at carry height (joint space, hand to preshape) -> descend (Cartesian) -> close -> lift -> carry
  above the bin (Cartesian, else joint space direct / via a raised midpoint) -> lower to drop height -> release
  (hand.release) -> retreat up -> home (hand back to open)
Each side is tried with every grasp.finger_hints direction (fingers forward can touch the centre bin's wall).
Cross-body (neither arm can do object -> bin): a RELOCATION plan picks with one arm and puts the object down on
the counter at a spot from which the bin-side arm has a full plan; the policy's next look picks it up from there.
Every planned tick is checked before it is used:
  * per-joint speed <= motion.arm_max_vel (trapezoidal time scaling on the inf-norm path length),
  * hand points (palm, grasp point, finger links) over the counter footprint stay above the counter
    (counter_clearance in transit; fingertip_clearance during the grasp itself),
  * hand points over a bin footprint (+1 cm) stay above its rim, except while releasing,
  * a dry run through a fresh SafetyFilter (same config as the env) produces 0 clipped ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter
from aaltonomous.interface.tasks import FIXED_MAPPING

from . import grasp_select
from .instruction import target_bin
from .kinematics import ArmKinematics


@dataclass
class Plan:
    ok: bool
    side: str | None = None
    color: str = ""
    reason: str = ""
    target_bin: str | None = None
    waypoints: dict = field(default_factory=dict)      # name -> grasp point (pelvis)
    ik: dict = field(default_factory=dict)             # name -> (ok, pos_err, tilt_deg)
    arm: np.ndarray | None = None                      # (T, 14)
    hand: np.ndarray | None = None                     # (T, 14)
    phases: list = field(default_factory=list)         # (name, start tick, end tick)
    safety_clips: int | None = None
    det: object = None                                 # the Detection this plan was made for (set by the policy)
    grasp: dict = field(default_factory=dict)          # grasp_select.select() result used for this plan

    @property
    def duration(self):
        return 0.0 if self.arm is None else len(self.arm) / 30.0

    def summary(self):
        if not self.ok:
            return f"{str(self.color):<6} SKIP ({self.reason})"
        wp = " ".join(f"{k}=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for k, v in self.waypoints.items()
                      if k in ("grasp", "drop"))
        tilt = max(t for _, _, t in self.ik.values())
        err = max(e for _, e, _ in self.ik.values()) * 1000
        return (f"{str(self.color):<6} {self.side:<5} arm -> {self.target_bin} bin  {wp}  IK ok max_err={err:.1f}mm "
                f"max_tilt={tilt:.0f}deg  {self.duration:.1f}s  safety_clips={self.safety_clips}")


def time_scale(Qs, vmax, amax, dt):
    """Piecewise-linear joint path through Qs (n, d) -> samples (T, d) at dt, EXCLUDING Qs[0], with a trapezoidal
    profile on the inf-norm arc length: no joint ever moves faster than vmax."""
    Qs = np.asarray(Qs, float)
    seg = np.abs(np.diff(Qs, axis=0)).max(axis=1)
    S = float(seg.sum())
    if S < 1e-9:
        return np.empty((0, Qs.shape[1]))
    ta = vmax / amax
    if amax * ta * ta > S:                       # triangle profile
        ta = np.sqrt(S / amax)
        vpk = amax * ta
        T = 2 * ta
    else:
        vpk = vmax
        T = 2 * ta + (S - amax * ta * ta) / vmax
    n = max(1, int(np.ceil(T / dt)))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * amax * t ** 2,
                 np.where(t <= T - ta, 0.5 * amax * ta ** 2 + vpk * (t - ta), S - 0.5 * amax * (T - t) ** 2))
    s = np.clip(s, 0, S)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((n, Qs.shape[1]))
    for k, sk in enumerate(s):
        i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-12 else (sk - cum[i]) / seg[i]
        out[k] = Qs[i] + f * (Qs[i + 1] - Qs[i])
    return out


def ramp(a, b, vmax, dt):
    """Linear move a -> b (excluding a) with every component at most vmax."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.ceil(np.abs(b - a).max() / (vmax * dt) + 1e-9))
    if n == 0:
        return np.empty((0, len(a)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


def _tri_shift(phi, dims=None, r=0.030):
    """Midpoint (m, along f) of an equilateral triangle's extent in direction f, f at angle phi from the apex direction;
    vertices at r from the centroid (altitude 45 mm: apex 30 mm, faces 15 mm). phi=0 -> +7.5 mm (towards the apex)."""
    if dims is not None and "circumradius" in dims:
        r = float(dims["circumradius"])
    v = np.radians([0.0, 120.0, 240.0])
    return 0.5 * (r * np.cos(phi - v).max() - r * np.cos(phi + np.pi - v).max())


class Planner:
    def __init__(self, cfg, scene, postures, home, kin: ArmKinematics | None = None, safety: SafetyConfig | None = None,
                 spec: JointSpec | None = None, mapping=None, dt=1.0 / 30.0):
        self.cfg, self.scene, self.post, self.home = cfg, scene, postures, np.asarray(home, float)
        self.spec = spec or JointSpec()
        self.kin = kin or ArmKinematics(cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=dt)
        self.dt = dt
        self.height = float(cfg["perception"].get("object_height") or scene.object_height())
        self.mapping = mapping or dict(FIXED_MAPPING)  # sort_fixed: fixed colour -> bin position
        gs = cfg["grasp"].get("select") or {}
        self.select_cfg = gs if gs.get("enabled") else None
        self.hand_geom = grasp_select.hand_geometry(self.kin, postures, gs.get("pad_width", 0.018)) if self.select_cfg else None

    def select_grasp(self, det):
        """Grasp type + lateral offset + height for this detection (grasp_select.py); None = configured point."""
        if self.select_cfg is None:
            return None
        shape = getattr(det, "shape", None)
        dims = self.scene.object_dims(shape) if shape else {"width": self.scene.size, "height": self.height,
                                                            "fillet": float(self.scene.cfg["objects"]["fillet"])}
        return grasp_select.select(shape, dims, self.hand_geom, self.select_cfg)

    # ------------------------------------------------------------------ geometry helpers
    def bin_xy(self, name):
        return self.scene.mat_to_pelvis(np.r_[self.scene.cfg["bins"]["centers"][name], 0.0])[:2]

    def drop_xy(self, name):
        """Release point (pelvis xy) in bin `name`: the point of the inner footprint nearest the pelvis, `grasp.drop_inset`
        inside the inner walls (the published box row puts the box centres past the arm's reach). None -> centre.
        VARIANT `grasp.drop_spread` (list of [dx, dy], off = null): offset cycled by self.drop_count[name] (objects already
        released into that bin), so a second object doesn't land on the first and poke above the rim."""
        xy = self._drop_xy(name)
        spread = self.cfg["grasp"].get("drop_spread")
        if spread:
            k = getattr(self, "drop_count", {}).get(name, 0)
            xy = xy + np.asarray(spread[k % len(spread)], float)
        return xy

    def _drop_xy(self, name):
        inset = self.cfg["grasp"].get("drop_inset")
        c = self.bin_xy(name)
        if inset is None:
            return c
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        lo2 = np.minimum(np.minimum(a, b) + inset, c)
        hi2 = np.maximum(np.maximum(a, b) - inset, c)
        return np.clip(np.zeros(2), lo2, hi2)

    def _over_counter(self, p):
        pm = self.scene.pelvis_to_mat(p)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        return np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)

    def bin_clearance(self, side, arm_q, hand_q, margin=0.01):
        """Lowest height (m) above the bin rims of the hand points over a bin's outer footprint (+margin), inf if none.
        Bins are static boxes in the sim and on the table: only the drop itself may go below a rim."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + margin
        rim = self.scene.counter_z + float(b["wall_height"])
        out = np.inf
        for c in b["centers"].values():
            over = np.all(np.abs(pm[:, :2] - np.asarray(c, float)) <= half, axis=1)
            if over.any():
                out = min(out, float((pts[over, 2] - rim).min()))
        return out

    def clearance(self, side, arm_q, hand_q, fingers_only=False):
        """Lowest height (m) above the counter surface of the hand points that are over the counter (inf if none)."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        if fingers_only:
            pts = pts[2:]
        over = self._over_counter(pts)
        if not over.any():
            return np.inf
        return float((pts[over, 2] - self.scene.counter_z).min())

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, side, p, seed, name, plan):
        q, ok, err, tilt = self.kin.solve(side, p, seed, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        plan.ik[name] = (ok, err, float(np.degrees(tilt)))
        plan.waypoints[name] = np.asarray(p, float)
        return q, ok

    def _cartesian(self, side, p0, p1, q0, name, plan):
        """IK waypoints along p0 -> p1 (see _cartesian_line). VARIANTS (motion section, off by default):
        `carry_joint_first`: the carry is a joint-space move (direct / raised midpoint) first, Cartesian as fallback;
        `shortcut_tol` (m): a Cartesian segment becomes ONE joint-space move when the grasp point stays this close to the
        straight line (per-waypoint IK lets the wrist wander: longer joint paths = slower at the same joint speed cap).
        Clearance + safety checks run on the final trajectory either way (oracle.py's _OraclePlanner, ported)."""
        mo = self.cfg["motion"]
        if name == "carry" and mo.get("carry_joint_first"):
            wall = float(self.scene.cfg["bins"]["wall_height"])
            z_min = self.scene.counter_z + wall + self.height / 2 + self.cfg["grasp"]["drop_above_rim"]
            reason = plan.reason
            out = self._joint_carry(side, q0, p0, p1, z_min, "carry", plan)
            if out is not None:
                return out
            plan.reason = reason
        qs = self._cartesian_line(side, p0, p1, q0, name, plan)
        tol = mo.get("shortcut_tol")
        if not tol or not qs or len(qs) < 2:
            return qs
        q0, q1 = np.asarray(q0, float), qs[-1]
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L2 = float(d @ d)
        for f in np.linspace(0, 1, 12)[1:-1]:
            p = self.kin.grasp_pose(side, q0 + f * (q1 - q0))[0]
            t = np.clip((p - p0) @ d / L2, 0, 1) if L2 > 0 else 0.0
            if np.linalg.norm(p - (p0 + t * d)) > tol:
                return qs
        return [q1]

    def _cartesian_line(self, side, p0, p1, q0, name, plan):
        """IK waypoints along the straight line p0 -> p1 (excluding p0), seeded continuously. A step whose joints
        jump more than `max_joint_step` is bisected (up to 3 times) before it counts as a failure."""
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / self.cfg["motion"]["cart_step"])))
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qs, q, info = [], q0, [True, 0.0, 0.0]
        jmax = float(self.cfg["motion"].get("max_joint_step", 0.4))

        def seg(pa, pb, q, depth):
            qn, ok, err, tilt = self.kin.solve(side, pb, q, top_down=True, restarts=False)
            if ok and np.abs(qn - q).max() <= jmax:
                info[1], info[2] = max(info[1], err), max(info[2], float(np.degrees(tilt)))
                return [qn]
            if depth >= 3:
                plan.reason = plan.reason or (f"IK jump on {name}" if ok else "")
                return None
            pm = 0.5 * (pa + pb)
            a = seg(pa, pm, q, depth + 1)
            b = a and seg(pm, pb, a[-1], depth + 1)
            return a + b if b else None

        for k in range(1, n + 1):
            out = seg(p0 + (p1 - p0) * (k - 1) / n, p0 + (p1 - p0) * k / n, q, 0)
            if out is None:
                plan.ik[name] = (False, info[1], info[2])
                return None
            qs += out
            q = out[-1]
        plan.ik[name] = (True, info[1], info[2])
        plan.waypoints[name] = p1
        return qs

    def _joint_carry(self, side, q_from, p_from, p_to, z_min, name, plan):
        """Joint-space move to an IK solution at p_to (direct, else via a raised midpoint), accepted if the grasp
        point never dips below z_min on the way."""
        def low(qa, qb):
            return any(self.kin.grasp_pose(side, qa + f * (qb - qa))[0][2] < z_min for f in np.linspace(0, 1, 15))

        q_to, ok = self._ik(side, p_to, q_from, name, plan)
        if ok and not low(q_from, q_to):
            return [q_to]
        for dz in (0.06, 0.12):
            mid = 0.5 * (np.asarray(p_from, float) + p_to) + [0, 0, dz]
            q_mid, ok1 = self._ik(side, mid, q_from, name + "_mid", plan)
            if not ok1 or low(q_from, q_mid):
                continue
            q_to, ok2 = self._ik(side, p_to, q_mid, name, plan)
            if ok2 and not low(q_mid, q_to):
                plan.reason = ""
                return [q_mid, q_to]
        plan.ik[name] = (False, 0.0, 0.0)
        plan.reason = f"{name}: no joint-space path above the bin rim"
        return None

    # ------------------------------------------------------------------ main
    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        """Plan for Detection `det` starting from the commanded arm_q/hand_q (14,). Tries the preferred arm (the
        target bin's side; for the centre bin the object's side), then the other one. If neither arm can do the
        whole pick -> bin (cross-body: the arm that reaches the object cannot reach the bin), plans a RELOCATION
        instead (`grasp.relocate`): pick + place on the counter at a spot from which an arm can finish the job;
        the next look re-detects the object there. obstacles: (n, 2) pelvis xy of the other detected objects.
        Returns the first Plan that passes every check, else the last failure."""
        g = self.cfg["grasp"]
        bin_name = bin_name or target_bin(self.mapping, det.color, getattr(det, "shape", None))
        by = self.bin_xy(bin_name)[1] if bin_name is not None else 0.0
        ref = by if abs(by) > 0.1 else det.top[1]
        side0 = "left" if ref > g["y_split"] else "right"
        last = Plan(False, color=det.color, reason=f"no bin for colour {det.color!r}")
        if bin_name is None:
            return last
        reasons, good = [], []
        for side in (side0, "right" if side0 == "left" else "left"):
            p = self._plan_side_hints(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float))
            if p.ok:
                if g.get("arm", "best") != "best":
                    return p
                good.append(p)
            else:
                reasons.append(f"{side}: {p.reason}")
                last = p
        if good:
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons)
        if relocate and g.get("relocate", {}).get("enabled", True):
            r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles)
            if r.ok:
                return r
            last.reason += f"; relocate: {r.reason}"
        return last

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Candidate counter spots (pelvis xy) for a relocation, nearest the target bin first."""
        rc = self.cfg["grasp"].get("relocate", {})
        sc = self.scene
        L = sc.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 4)
        pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        return keep[: int(rc.get("max_spots", 6))]

    def plan_relocation(self, det, bin_name, arm_q, hand_q, obstacles=()):
        """Cross-body fallback: pick `det` with an arm that reaches it and put it down at a counter spot from which
        some arm has a full plan to `bin_name` (checked from home). Returns a Plan with target_bin 'counter'."""
        from dataclasses import replace
        last = Plan(False, color=det.color, reason="no relocation spot")
        H = self.height
        for spot in self.relocation_spots(det, bin_name, obstacles):
            d2 = replace(det, top=np.r_[spot, self.scene.counter_z + H])
            by = self.bin_xy(bin_name)[1]
            fin_side = "left" if (by if abs(by) > 0.1 else spot[1]) > 0 else "right"
            fin = self._plan_side_hints(d2, fin_side, bin_name, self.home.copy(), self.post["open"].copy())
            if not fin.ok:
                last.reason = f"no {fin_side} plan from spot ({spot[0]:+.2f},{spot[1]:+.2f}): {fin.reason}"
                continue
            side = "right" if fin_side == "left" else "left"
            p = self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=spot)
            if p.ok:
                p.reason = f"relocate for the {fin_side} arm"
                return p
            last = p
            last.reason = f"{side} cannot place at ({spot[0]:+.2f},{spot[1]:+.2f}): {p.reason}"
        return last
    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        """_plan_side with each finger direction of grasp.finger_hints in turn (fingers forward can reach over the
        centre bin's wall when the object is close to it; then fingers inward/sideways). VARIANT `grasp.face_align`
        (needs det.shape + det.yaw): face-aligned directions first, the hints as the fallback."""
        fa = self.cfg["grasp"].get("face_align") or {}
        if fa.get("enabled") and getattr(det, "shape", None) in (fa.get("shapes") or {}) and getattr(det, "yaw", None) is not None:
            p = self._plan_side_facealign(det, side, bin_name, arm_q, hand_q, fa, **kw)
            if p is not None:
                return p
        return self._plan_side_finger_hints(det, side, bin_name, arm_q, hand_q, **kw)

    def _plan_side_facealign(self, det, side, bin_name, arm_q, hand_q, fa, **kw):
        """Fingers along a direction fixed to the object's faces (oracle.py's face-aligned grasp, generalised):
        f = yaw + offset + k * 360/n deg, n = rotational symmetry (cube 4, triangle 3). Offsets in the scene/export.py
        mesh frame: cube faces at 0/90/..; triangle apex along +y, so offset 90 + phi = fingers phi deg from the apex
        direction (phi 0: fingers straddle the apex = the two fingers on the two faces meeting there, thumb on the
        opposite flat face: three faces; phi 180: thumb on the apex, fingers on one face). Candidates nearest the side's
        natural finger direction first; the achieved direction must be within tol_deg. None = no face plan."""
        import copy
        shape, yaw = det.shape, float(det.yaw)
        sh = fa["shapes"][shape]
        off, n = np.radians(float(sh.get("offset_deg", 0.0))), int(sh.get("n", 4))
        ref = np.radians(float((fa.get("finger_ref_deg") or {"left": -25.0, "right": 25.0})[side]))
        max_dev, tol = np.radians(float(fa.get("max_dev_deg", 75.0))), float(fa.get("tol_deg", 12.0))
        cands = []
        for k in range(n):
            a = yaw + off + 2 * np.pi * k / n
            da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
            if da <= max_dev:
                cands.append((da, a))
        saved = (self.kin.hint, self.kin.w_hint)
        try:
            for _, a in sorted(cands):
                f = np.array([np.cos(a), np.sin(a), 0.0])
                d2 = copy.copy(det)
                if shape == "triangle" and sh.get("support_shift", True):   # pinch midpoint along f != centroid
                    d2.top = det.top + _tri_shift(a - (yaw + np.pi / 2), self.scene.object_dims("triangle")) * f
                self.kin.hint, self.kin.w_hint = f, float(fa.get("hint_weight", 3.0))
                p = self._plan_side(d2, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    err = self._finger_err(p, side, f)
                    if err <= tol:
                        p.reason = f"face-aligned ({np.degrees(a):.0f} deg, err {err:.0f})"
                        return p
        finally:
            self.kin.hint, self.kin.w_hint = saved
        return None

    def _finger_err(self, plan, side, f):
        """Angle (deg) between the palm x axis (finger direction) at the grasp and f, in the horizontal plane."""
        e = next(e for name, s, e in plan.phases if name == "descend")
        self.kin._fk(plan.arm[e - 1])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        a = np.arctan2(x[1], x[0]) - np.arctan2(f[1], f[0])
        return float(np.degrees(abs((a + np.pi) % (2 * np.pi) - np.pi)))

    def _plan_side_finger_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        hints = self.cfg["grasp"].get("finger_hints") or [self.cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0])]
        # Retry of the same object (policy sets hint_shift = earlier attempts at this spot): start with the next
        # finger direction, so a retry is not the identical grasp that just failed (`grasp.retry_rotate_hints`).
        shift = int(getattr(self, "hint_shift", 0)) % len(hints) if self.cfg["grasp"].get("retry_rotate_hints") else 0
        hints = list(hints[shift:]) + list(hints[:shift])
        saved, first = self.kin.hint, None
        try:
            for h in hints:
                h = np.asarray(h, float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
                self.kin.hint = h / np.linalg.norm(h)
                p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    return p
                first = first or p
            return first
        finally:
            self.kin.hint = saved

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None):
        """VARIANT `grasp.per_shape.<shape>` (needs det.shape): {point_in_palm, height_frac, preshape, closed (posture
        names in `hand`), close_hold_s, lift_vel} for that shape only."""
        ps = dict((self.cfg["grasp"].get("per_shape") or {}).get(getattr(det, "shape", None)) or {})
        saved = dict(self.kin.offset)
        sel = self.select_grasp(det)
        if ps.get("point_in_palm") is not None:            # explicit per-shape point wins over the selection
            p = np.asarray(ps["point_in_palm"], float)
            self.kin.offset = {"left": p, "right": p * np.array([1.0, -1.0, 1.0])}
        elif sel is not None:                               # lateral (palm z) from the geometry, x / y as configured
            p = np.asarray(saved["left"], float).copy()
            p[2] = self.hand_geom["centre"] + self.hand_geom["pinch_sign"] * sel["lateral"]
            self.kin.offset = {"left": p, "right": p * np.array([1.0, -1.0, 1.0])}
            ps.setdefault("height_frac", sel["height_frac"])
        try:
            plan = self._plan_side_impl(det, side, bin_name, arm_q, hand_q, place_xy, ps)
            plan.grasp = sel or {}
            return plan
        finally:
            self.kin.offset = saved

    def _plan_side_impl(self, det, side, bin_name, arm_q, hand_q, place_xy, ps):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter")
        sl = ArmKinematics.sl(side)
        shape = getattr(det, "shape", None)
        H, post = (self.scene.object_height(shape) if shape else self.height), self.post   # this object's own height
        hand_pre = hand_q.copy(); hand_pre[sl] = post[ps.get("preshape", "preshape")][sl]
        hand_cls = hand_q.copy(); hand_cls[sl] = post[ps.get("closed", "closed")][sl]
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]

        # grasp height: object mid-height, raised until the preshaped fingers clear the counter
        xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
        z = zc + float(ps.get("height_frac", g["height_frac"])) * H
        for _ in range(4):
            qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
            if not ok:
                plan.reason = f"IK grasp {side}"
                return plan
            c = self.clearance(side, qg, hand_pre, fingers_only=True)
            if c >= g["fingertip_clearance"] - 1e-3:
                break
            z += g["fingertip_clearance"] - c
        else:
            plan.reason = "fingers hit the counter"
            return plan
        if z > zc + H - 0.008:
            plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
            return plan

        wall = float(sc.cfg["bins"]["wall_height"])
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        z_drop = zc + wall + H / 2 + g["drop_above_rim"]
        bxy = self.drop_xy(bin_name)
        if place_xy is not None:  # relocation: put it down on the counter (release just above the grasp height)
            bxy, z_drop = np.asarray(place_xy, float), z + 0.005
        P = {"pregrasp": np.r_[xy, max(z + g["approach_height"], z_carry)], "grasp": np.r_[xy, z], "lift": np.r_[xy, z_carry],
             "above_bin": np.r_[bxy, z_carry], "drop": np.r_[bxy, z_drop], "retreat": np.r_[bxy, z_drop + g["retreat_up"]]}

        q_pre, ok = self._ik(side, P["pregrasp"], qg, "pregrasp", plan)  # same IK branch as the grasp
        if not ok:
            plan.reason = f"IK pregrasp {side}"
            return plan
        plan.waypoints["grasp"] = np.r_[xy, z]
        down = self._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
        up = down and self._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
        carry = up and (self._cartesian(side, P["lift"], P["above_bin"], up[-1], "carry", plan)
                        or self._joint_carry(side, up[-1], P["lift"], P["above_bin"], z_drop, "carry", plan))
        lower = carry and self._cartesian(side, P["above_bin"], P["drop"], carry[-1], "lower", plan)
        retreat = lower and self._cartesian(side, P["drop"], P["retreat"], lower[-1], "retreat", plan)
        if not retreat:
            plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            return plan
        plan.waypoints["drop"] = P["drop"]

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def move(name, Qs, hand_to=None, v=None):
            nonlocal cur_a, cur_h
            a = time_scale([cur_a] + list(Qs), v or vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:  # hand moves during the arm move (ends no later than the arm)
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            h = ramp(cur_h, to, hv, self.dt)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        move("to_pregrasp", [q_pre], hand_to=hand_pre)
        move("descend", down)
        hand("close", hand_cls, float(ps.get("close_hold_s", mo["close_hold_s"])))
        move("lift", up, v=ps.get("lift_vel", mo.get("lift_vel")))   # VARIANT (per_shape.)lift_vel, null = arm_max_vel
        move("carry", carry + lower)
        hand("open", hand_rel, mo["open_hold_s"])  # widest posture: `open` alone left the cube wedged in the sim
        move("retreat", retreat)
        move("home", [home], hand_to=hand_open)
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        grasp_ticks = np.zeros(len(plan.arm), bool)
        near = ("descend", "close", "lift") + (("carry", "open", "retreat") if place_xy is not None else ())
        for name, s, e in phases:
            if name in near:
                grasp_ticks[s:e] = True
        closing = np.zeros(len(plan.arm), bool)
        for name, s, e in phases:
            closing[s:e] |= name == "close" or (place_xy is not None and name == "open")
        for k in range(0, len(plan.arm), 3):
            # closing: the thumb swings through its lowest point; touching the mat lightly is acceptable
            need = 0.003 if closing[k] else g["fingertip_clearance"] if grasp_ticks[k] else mo["counter_clearance"]
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                # the object itself sits between the fingers during descend/lift; only the counter matters here
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
                return plan
        drop = np.zeros(len(plan.arm), bool)
        for name, s_, e in phases:
            drop[s_:e] |= name in ("open",)
        for k in range(0, len(plan.arm), 3):
            if drop[k]:
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                name = next(n for n, s_, e in phases if s_ <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan

    def dry_run_safety(self, arm_q, hand_q, arm, hand):
        f = SafetyFilter(self.spec, self.safety)
        f.reset(arm_q, hand_q, 0.0)
        for k in range(len(arm)):
            f.filter({"arm_q_target": arm[k], "hand_q_target": hand[k]}, (k + 1) * self.dt)
        return f.clipped_steps

    def go_to(self, arm_from, hand_from, arm_to, hand_to):
        """Joint-space move (both arms + hands) at the configured speeds, e.g. start pose -> home."""
        mo = self.cfg["motion"]
        a = time_scale([arm_from, arm_to], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = ramp(hand_from, hand_to, mo["hand_max_vel"], self.dt)
        n = max(len(a), len(h))
        a = np.vstack([a, np.repeat(np.asarray(arm_to, float)[None], n - len(a), 0)]) if len(a) < n else a
        h = np.vstack([h, np.repeat(np.asarray(hand_to, float)[None], n - len(h), 0)]) if len(h) < n else h
        return a, h
