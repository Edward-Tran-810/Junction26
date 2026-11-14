"""Finger-load grasp monitor: is something between the fingers?

Signal: the Dex3 joints are position controlled, so an object between the fingers stops them short of the commanded
posture. The steady-state tracking error |measured - commanded| is the only contact feedback the interface gives
(`hand_q` in every observation; no tactile, no torque). Measured in the Aalto Isaac sim on 24 classical picks
(runs/eval/classical_v2percep_sort_fixed_nominal_20261114_175410, >= 1.0 s after the closed posture was commanded,
until the release):

    object held (17 picks):  thumb joints >= 0.022 rad, finger joints >= 0.029 rad, sum >= 0.051 rad
    hand empty  ( 7 picks):  thumb joints  = 0.000 rad, finger joints <= 0.006 rad, sum <= 0.006 rad

Two independent joint groups (thumb = 3 joints, index + middle = 4 joints, different actuators) must BOTH carry load
for `held`, and a verdict needs `confirm_samples` consecutive agreeing samples, so one noisy reading or one finger
resting on something is not proof of a grasp. Readings are only taken while the commanded posture has been constant
for `settle_s` (a moving target has tracking lag that looks like load).

After the release the same reading, taken in the half-bent `open` posture, tells a clean release (0.000-0.001 rad)
from an object still hooked / wedged in the fingers (0.035 and 0.052 rad on the two wedged objects of
runs/eval/rl_v2percep_sort_fixed_nominal_20261114_181306, seed 1000). In the straight-finger `release` posture the
reading is useless (0.13 rad with an empty hand: the sim fingers stop at -0.08 rad, short of the commanded -0.05).

All thresholds are SIM-MEASURED (kp 100 hand). The real Dex3 controller is ~67x softer (kp 1.5): expect larger
errors both held and empty -> re-measure `verify.*` on the robot before trusting it there (verify.enabled: false
restores the open-loop behaviour).
"""
from __future__ import annotations

import numpy as np

HZ = 30.0
THUMB, FINGERS = slice(0, 3), slice(3, 7)     # per-hand joint groups (both hands: thumb first, then two fingers)


def side_slice(side):
    return slice(0, 7) if side == "left" else slice(7, 14)


def finger_load(hand_q, hand_cmd, side):
    """(thumb load, finger load) in rad: L1 norm of measured - commanded over each joint group of one hand."""
    sl = side_slice(side)
    e = np.abs(np.asarray(hand_q, float)[sl] - np.asarray(hand_cmd, float)[sl])
    return float(e[THUMB].sum()), float(e[FINGERS].sum())


class GraspMonitor:
    """Feed every emitted hand command to track() and every observation to sample()."""

    def __init__(self, cfg):
        v = cfg.get("verify") or {}
        self.enabled = bool(v.get("enabled", False))
        self.settle_ticks = int(round(float(v.get("settle_s", 1.0)) * HZ))
        self.thumb_held = float(v.get("thumb_held", 0.010))
        self.fingers_held = float(v.get("fingers_held", 0.014))
        self.load_empty = float(v.get("load_empty", 0.010))
        self.confirm = max(1, int(v.get("confirm_samples", 2)))
        self.confirm_ticks = int(round(float(v.get("confirm_s", 0.2)) * HZ))   # ... spanning at least this long
        r = v.get("release") or {}
        self.release_settle_ticks = int(round(float(r.get("settle_s", 0.5)) * HZ))
        self.load_stuck = float(r.get("load_stuck", 0.015))
        self.reset()

    def reset(self):
        self._last = None                      # last emitted hand command (14,)
        self.steady = {"left": 0, "right": 0}  # ticks the side's commanded posture has been constant
        self._run = (None, 0, 0)               # (raw verdict, consecutive readings, ticks they span)
        self.clock = 0                         # ticks emitted so far
        self._t_sample = 0
        self.last_load = (0.0, 0.0)

    def restart(self):
        """Forget the running verdict (new verification window)."""
        self._run = (None, 0, 0)
        self._t_sample = self.clock

    # ------------------------------------------------------------------ commands
    def track(self, hand_chunk):
        """hand_chunk: (n, 14) hand targets just handed to the runner."""
        h = np.atleast_2d(np.asarray(hand_chunk, float))
        if not len(h):
            return
        self.clock += len(h)
        full = h if self._last is None else np.vstack([self._last[None], h])
        for side in ("left", "right"):
            x = full[:, side_slice(side)]
            moved = np.flatnonzero(np.abs(np.diff(x, axis=0)).max(axis=1) > 1e-6) if len(x) > 1 else np.array([], int)
            if len(moved):
                self.steady[side] = len(x) - 1 - (int(moved[-1]) + 1)   # ticks after the last change
            else:
                self.steady[side] += len(h)
        self._last = h[-1].copy()

    # ------------------------------------------------------------------ observations
    def classify(self, thumb, fingers):
        if thumb >= self.thumb_held and fingers >= self.fingers_held:
            return "held"
        if thumb + fingers <= self.load_empty:
            return "empty"
        return "unsure"

    def sample(self, obs, hand_cmd, side):
        """One reading. Returns "held" | "empty" (confirmed: `confirm_samples` consecutive readings that agree and span
        >= confirm_s), "unsure" (load between the thresholds, or not confirmed yet) or "wait" (commanded posture not
        constant for settle_s yet)."""
        if self.steady[side] < self.settle_ticks:
            return "wait"
        th, fi = finger_load(obs["hand_q"], hand_cmd, side)
        self.last_load = (th, fi)
        raw = self.classify(th, fi)
        prev, n, span = self._run
        dt, self._t_sample = self.clock - self._t_sample, self.clock
        n, span = (n + 1, span + dt) if raw == prev else (1, 0)
        self._run = (raw, n, span)
        if raw == "unsure" or n < self.confirm or span < self.confirm_ticks:
            return "unsure"
        return raw

    def release_reading(self, obs, hand_cmd, side):
        """After the release, in the `open` posture: "clean" | "stuck" | "wait" (posture not settled yet)."""
        if self.steady[side] < self.release_settle_ticks:
            return "wait"
        th, fi = finger_load(obs["hand_q"], hand_cmd, side)
        self.last_load = (th, fi)
        return "stuck" if th + fi >= self.load_stuck else "clean"
