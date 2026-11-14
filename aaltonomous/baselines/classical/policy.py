"""ClassicalPolicy: obs -> action wrapper around perception + planner (rung 1, sort_fixed).

    pol = ClassicalPolicy()                   # config.yaml, scene.yaml, calibration.yaml
    pol.reset({"instruction": obs["instruction"]})
    while True:
        action = pol.act(obs)                 # {"arm_q_target": (k,14), "hand_q_target": (k,14), "done": bool}
        obs, info = env.step(action)

Behaviour: go home -> settle -> look (head RGB-D) -> plan one object -> execute -> back home -> look again ...
until nothing sortable is left (then `done`). While it is planning (background thread when `threaded=True`) or
waiting for an operator confirm, act() returns 1-tick HOLD actions (the last target), so the env's watchdog
never fires. Targets are absolute, planned at <= motion.arm_max_vel and pre-checked against a SafetyFilter.

Operator confirm (real robot): ClassicalPolicy(require_confirm=True) asks before the first move to home and before
each object; the runner calls confirm() / skip() / stop() while `pol.waiting` is True (`pol.pending_text` says what).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .instruction import mapping_for_task, target_bin
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Planner

HZ = 30.0


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


class ClassicalPolicy:
    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        # Optional RL grasp residual (aaltonomous/rl/skill.py). OFF unless cfg["rl_grasp"] or $AALTO_RL_GRASP
        # names an exported actor; with it off nothing below changes.
        rl = self.cfg.get("rl_grasp") or os.environ.get("AALTO_RL_GRASP")
        self.skill = None
        if rl:
            from aaltonomous.rl.skill import GraspSkill
            self.skill = GraspSkill(rl, self.spec)
        self.require_confirm = require_confirm
        self.threaded = threaded
        if threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        if getattr(self, "perception", None) is not None:   # new episode: re-estimate the camera correction
            self.perception.reset_calibration()               # on its first frame (perception.refine_cache)
        if info.get("mapping") is not None:      # given (oracle / tests); an empty mapping means "sort nothing"
            self.mapping = dict(info["mapping"])
        elif info.get("instruction") and info.get("task") != "sort_fixed":
            # rung 2: parse the text. Unclear parts stay unmapped (left on the counter), never guessed.
            self.mapping = mapping_for_task(info.get("task"), info["instruction"])
        else:
            self.mapping = dict(FIXED_MAPPING)
        self.planner.mapping = self.mapping
        self.time_limit = info.get("time_limit", self.time_limit)
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.attempts = []          # (position, count) of objects we tried to pick
        self.failed = []            # positions whose plans failed
        self.pending = None         # Plan waiting for the operator
        self.pending_text = ""      # what the operator is asked to confirm
        self.waiting = False
        self.events = []
        self.plans = []
        self._job = None
        self._last_place = None
        self.place_checks = []
        self._decision = None
        self.last_detections = []
        self._last_place = None     # (grasp xy, colour, shape, target bin, detections before the pick)
        self.place_checks = []      # A5 re-look: one outcome per executed pick (see _check_last_place)

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        if self.arm_cmd is None:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if self.queue_a is not None and len(self.queue_a):
            if self.skill is not None and self.skill.active:
                return self.skill.act(self, obs)
            return self._pop()
        if self.state == "init":
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._home_move = (a, h)
            dq = np.abs(self.home - self.arm_cmd).max()
            self.pending_text = f"move both arms to home_arm_q (max joint change {dq:.2f} rad, {len(a) / HZ:.1f} s)"
            if self.require_confirm:
                self.state, self.waiting, self.pending = "await_home", True, None
                self._note(t, f"waiting for confirm: {self.pending_text}")
                return self._hold()
            self.state = "home"
        if self.state == "await_home":
            d, self._decision = self._decision, None
            if d in ("skip", "stop"):
                self.waiting, self.state = False, "done"
                self._note(t, "operator stop before the first move")
                return self._hold(done=True)
            if d != "go":
                return self._hold()
            self.waiting, self.state = False, "home"
        if self.state == "home":
            a, h = self._home_move
            self._enqueue(a, h)
            self.state = "settle"
            self._note(t, f"to home ({len(a) / HZ:.1f} s)")
            return self._pop() if len(a) else self._hold()
        if self.state == "settle":
            n = int(round(self.cfg["motion"]["settle_s"] * HZ))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "look"
            return self._pop()
        if self.state == "look":
            if self.threaded:
                if self._job is None:
                    self._job = {"obs": obs, "result": None}
                    threading.Thread(target=self._look_job, args=(self._job,), daemon=True).start()
                if self._job["result"] is None:
                    return self._hold()
                res, self._job = self._job["result"], None
            else:
                res = self._look(obs)
            return self._after_look(res, t)
        if self.state == "await":
            d, self._decision = self._decision, None
            if d == "go":
                self.waiting = False
                return self._start(self.pending, t)
            if d == "skip":
                self.waiting = False
                self.failed.append(self.pending.waypoints["grasp"][:2])
                self._note(t, f"operator skipped {self.pending.color}")
                self.state = "look"
                return self._hold()
            if d == "stop":
                self.waiting = False
                self.state = "done"
                self._note(t, "operator stop")
            return self._hold(done=self.state == "done")
        if self.state == "executing":  # queue drained -> back home: look again after settling
            self.state = "settle"
            return self.act(obs)
        return self._hold(done=True)

    # ------------------------------------------------------------------ look + plan
    def _look_job(self, job):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        try:
            job["result"] = self._look(job["obs"])
        except Exception as e:  # never kill the control loop
            job["result"] = ("error", repr(e))

    def _check_last_place(self, dets):
        """Re-look after a place (A5): where did the last picked object go?
          still_on_counter   an object of its colour is still within same_object_radius of the grasp point
                             (the grasp missed / slipped before lift): the normal loop retries it (max_attempts)
          dropped_elsewhere  not at the grasp point, but an object of its colour appeared on the counter where none
                             was before (fell out of the hand during the carry)
          left_counter       gone from the counter: in a bin (objects in bins are not reported by perception)
        Returns the outcome string (None if nothing was placed since the last look)."""
        if self._last_place is None:
            return None
        xy, col, shape, bin_name, before = self._last_place
        self._last_place = None
        r = self.cfg["policy"]["same_object_radius"]
        same = [d for d in dets if d.color == col and np.all(np.isfinite(d.top))]
        if any(np.linalg.norm(d.top[:2] - xy) < r for d in same):
            out = "still_on_counter"
        elif any(all(np.linalg.norm(d.top[:2] - b.top[:2]) >= r for b in before if b.color == col) for d in same):
            out = "dropped_elsewhere"
        else:
            out = "left_counter"
        self.place_checks.append({"color": col, "shape": shape, "bin": bin_name, "xy": [float(v) for v in xy],
                                  "outcome": out})
        return out

    def _look(self, obs):
        dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        self.last_detections = dets
        chk = self._check_last_place(dets)
        if chk is not None:
            self.log(f"[classical] re-look after place: {chk}")
        cands = []
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            n = sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < self.cfg["policy"]["same_object_radius"])
            if n >= self.cfg["policy"]["max_attempts"]:
                continue
            cands.append(d)
        if self.cfg["policy"]["order"] == "nearest_bin":
            def key(d):
                b = target_bin(self.mapping, d.color, getattr(d, "shape", None))
                return np.inf if b is None else np.linalg.norm(d.top[:2] - self.planner.bin_xy(b))
        else:
            def key(d):
                return -d.n_px
        cands.sort(key=key)
        for d in cands:
            others = [o.top[:2] for o in dets if o is not d]
            p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, obstacles=others)
            self.plans.append(p)
            if p.ok:
                return ("plan", p, len(dets))
            self.failed.append(d.top[:2])
            self.log(f"[classical] skip {d}: {p.reason}")
        return ("none", None, len(dets))

    def _after_look(self, res, t):
        kind = res[0]
        if kind == "error":
            self._note(t, f"look/plan error {res[1]}: stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "none":
            self._note(t, f"nothing left to sort ({res[2]} detections)")
            self.state = "done"
            return self._hold(done=True)
        plan = res[1]
        if t + plan.duration > self.time_limit:
            self._note(t, f"not enough time for {plan.color} ({plan.duration:.0f} s)")
            self.state = "done"
            return self._hold(done=True)
        if self.require_confirm:
            self.pending, self.waiting, self.state = plan, True, "await"
            self.pending_text = plan.summary()
            self._note(t, f"waiting for confirm: {plan.summary()}")
            return self._hold()
        return self._start(plan, t)

    def _start(self, plan, t):
        xy = plan.waypoints["grasp"][:2]
        self.attempts.append((xy, 1))
        det = getattr(plan, "det", None)
        self._last_place = (np.asarray(xy, float), plan.color, getattr(det, "shape", None), plan.target_bin,
                            list(self.last_detections))
        self._enqueue(plan.arm, plan.hand)
        if self.skill is not None:
            self.skill.begin(plan, self)
        self.state = "executing"
        self._note(t, f"execute {plan.summary()}")
        return self._pop()

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)

    def _pop(self):
        k = int(self.cfg["motion"]["chunk"])
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        if not len(a):
            return self._hold()
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _hold(self, done=False):
        """Hold the last target. While waiting (planning / operator) the hold is `motion.hold_ticks` ticks long, so a
        remote runner needs one request per hold chunk instead of one per tick: the command rate stays at 30 Hz
        even when the planner thread and other processes (sims) load the CPU."""
        k = 1 if done else max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
