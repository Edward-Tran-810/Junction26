"""Deployment side of the RL grasp: a residual on the classical plan's descend -> close -> lift ticks, CPU only.

Used by the classical pipeline only when enabled (off by default): ClassicalPolicy reads cfg["rl_grasp"] or
$AALTO_RL_GRASP (path to an exported TorchScript actor, `policy.pt`, with its meta json next to it).

Per tick inside the skill window the actor sees exactly what the training env's actor sees (grasp_env.py
_get_observations): measured arm_q / arm_dq / hand_q of the grasping side (right side mirrored into the left
frame), the plan's next target, the phase, the last action, the object estimate (shape, top xy, yaw) the plan was
made from. The command = plan + residual, projected onto (last command +- 97 % of the velocity cap) and the filter's
joint range, so nothing is clipped. After `lift` the arm residual decays to 0 at the remaining velocity headroom;
the hand keeps its last residual until the plan's `open` phase, then decays too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig

from .grasp_env import ACTION_DIMS, ARM_VEL_CAP, CAP_FRAC, HAND_VEL_CAP
from .scripted import HOLD_S, POST_S, SHAKE_S, SHAPES

ARM_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], float)
TAIL = int(round(HOLD_S * 30)) + int(round(SHAKE_S * 30)) + int(round(POST_S * 30))


def hand_r2l(r7):
    r = np.asarray(r7, float)
    return -np.array([r[0], r[1], r[2], r[5], r[6], r[3], r[4]])


hand_l2r = hand_r2l          # the map is its own inverse


class GraspSkill:
    def __init__(self, cfg, spec: JointSpec | None = None):
        import torch
        torch.set_num_threads(1)
        cfg = {"checkpoint": cfg} if isinstance(cfg, str) else dict(cfg)
        p = Path(cfg["checkpoint"])
        self.meta = json.loads((p.parent / "skill_meta.json").read_text())
        self.net = torch.jit.load(str(p), map_location="cpu").eval()
        self.torch = torch
        self.spec = spec or JointSpec()
        sc = SafetyConfig()
        self.lo = np.r_[self.spec.arm_lower + sc.arm_pos_margin, self.spec.hand_lower + sc.hand_pos_margin]
        self.hi = np.r_[self.spec.arm_upper - sc.arm_pos_margin, self.spec.hand_upper - sc.hand_pos_margin]
        self.cap = np.r_[[ARM_VEL_CAP * CAP_FRAC / 30] * 14, [HAND_VEL_CAP * CAP_FRAC / 30] * 14]
        e = self.meta["env"]
        self.dims = np.array(ACTION_DIMS[e.get("action_space_name", "full")])
        self.scale = np.r_[[e.get("residual_arm", 0.12)] * 7, [e.get("residual_hand", 0.5)] * 7][self.dims]
        self.use_est = bool(e.get("use_estimate", True))
        self.active = False
        self.times = []                     # per-tick inference wall time (s)

    # ------------------------------------------------------------------ plan hook
    def begin(self, plan, det=None):
        """Called when the classical policy starts executing `plan`. det: the Detection it was planned for."""
        ph = {n: (s, e) for n, s, e in plan.phases}
        if not all(k in ph for k in ("to_pregrasp", "descend", "close", "lift")):
            self.active = False
            return
        self.side = plan.side
        self.k = 0                                   # ticks of the plan already popped
        self.s0, self.e_lift = ph["to_pregrasp"][1], ph["lift"][1]
        self.e_open = ph["open"][0] if "open" in ph else len(plan.arm)
        self.starts = [0, ph["close"][0] - self.s0, ph["lift"][0] - self.s0]
        n = self.e_lift - self.s0
        self.starts += [n, n + 30, n + 60]
        self.L = n + TAIL
        self.last_a = np.zeros(len(self.dims))
        self.offset = np.zeros(28)                   # current residual (arm 14 + hand 14) of the last command
        sl = 1 if self.side == "right" else 0
        top = det.top if det is not None else plan.waypoints["grasp"]
        x, y = float(top[0]), float(top[1])
        yaw = float(getattr(det, "yaw", 0.0) or 0.0)
        if self.side == "right":
            y, yaw = -y, -yaw
        shape = getattr(det, "shape", None)
        oh = np.zeros(3)
        if shape in SHAPES:
            oh[SHAPES.index(shape)] = 1
        self.est = np.r_[oh, (x - 0.30) * 10, (y - 0.08) * 10, np.sin(yaw), np.cos(yaw), np.sin(3 * yaw),
                         np.cos(3 * yaw), np.sin(4 * yaw), np.cos(4 * yaw)]
        if not self.use_est:
            self.est[:] = 0
        self.active = True

    def _to_left(self, arm7, hand7):
        if self.side == "left":
            return np.asarray(arm7, float), np.asarray(hand7, float)
        return np.asarray(arm7, float) * ARM_MIRROR, hand_r2l(hand7)

    def _from_left(self, arm7, hand7):
        if self.side == "left":
            return arm7, hand7
        return arm7 * ARM_MIRROR, hand_l2r(hand7)

    # ------------------------------------------------------------------ per tick
    def act(self, pol, obs):
        """One action for ClassicalPolicy `pol` (its queue holds the rest of the plan)."""
        k = self.k
        if k < self.s0:                                        # before the window: normal chunks up to s0
            n = min(int(pol.cfg["motion"]["chunk"]), self.s0 - k)
            return self._pop_plain(pol, n)
        a_s, h_s = pol.queue_a[0].copy(), pol.queue_h[0].copy()
        sa = slice(0, 7) if self.side == "left" else slice(7, 14)
        res = np.zeros(14)
        if k < self.e_lift:
            t = k - self.s0
            arm_q, hand_q = self._to_left(np.asarray(obs["arm_q"])[sa], np.asarray(obs["hand_q"])[sa])
            arm_dq, _ = self._to_left(np.asarray(obs["arm_dq"])[sa], np.zeros(7))
            scr_a, scr_h = self._to_left(a_s[sa], h_s[sa])
            ph = int(np.searchsorted(self.starts, t, side="right") - 1)
            ph = min(max(ph, 0), 5)
            oh = np.zeros(6); oh[ph] = 1
            prog = np.array([t / max(self.L, 1), (t - self.starts[ph]) / 30.0])
            x = np.r_[arm_q, arm_dq, hand_q, scr_a, scr_h, oh, prog, self.last_a, self.est].astype(np.float32)
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                act = self.net(self.torch.from_numpy(x)[None])[0].numpy()
            self.times.append(time.perf_counter() - t0)
            act = np.clip(act, -1, 1)
            self.last_a = act
            res[self.dims] = act * self.scale
            ra, rh = self._from_left(res[:7], res[7:])
            off = np.zeros(28); off[sa] = ra; off[14:][sa] = rh
            self.offset = off
        else:                                                   # after lift: arm residual -> 0; hand until `open`
            self.offset[:14] = 0.0
            if k >= self.e_open:
                self.offset[14:] = 0.0
        desired = np.r_[a_s, h_s] + self.offset
        last = np.r_[pol.arm_cmd, pol.hand_cmd]
        tgt = np.clip(np.clip(desired, last - self.cap, last + self.cap), self.lo, self.hi)
        pol.queue_a, pol.queue_h = pol.queue_a[1:], pol.queue_h[1:]
        self.k += 1
        pol.arm_cmd, pol.hand_cmd = tgt[:14].copy(), tgt[14:].copy()
        if self.k >= self.e_lift and np.allclose(tgt, np.r_[a_s, h_s], atol=1e-6):
            self.active = False                                 # back on the plan: plain chunks again
        return {"arm_q_target": tgt[None, :14].copy(), "hand_q_target": tgt[None, 14:].copy(), "done": False}

    def _pop_plain(self, pol, n):
        a, h = pol.queue_a[:n], pol.queue_h[:n]
        pol.queue_a, pol.queue_h = pol.queue_a[n:], pol.queue_h[n:]
        self.k += len(a)
        pol.arm_cmd, pol.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}
