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
from .grasp_monitor import GraspMonitor, side_slice
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Planner, ramp, time_scale

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
        self.monitor = GraspMonitor(self.cfg)       # closed-loop grasp / transit / release checks (config `verify`)
        # Optional RL grasp residual (aaltonomous/rl/skill.py). OFF unless cfg["rl_grasp"] or $AALTO_RL_GRASP
        # names an exported actor; with it off nothing below changes.
        rl = self.cfg.get("rl_grasp") or os.environ.get("AALTO_RL_GRASP")
        self.skill = None
        if rl:
            from aaltonomous.rl.skill import GraspSkill
            if isinstance(rl, str):      # deployment envelope from config `rl_limits` ($AALTO_RL_LIMITS overrides)
                rl = {"checkpoint": rl, "limits": self.cfg.get("rl_limits") or {}}
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
        self.planner.drop_count = {}  # objects released per bin this episode (planner grasp.drop_spread variant)
        self.planner.hint_shift = 0
        if getattr(self, "monitor", None) is not None:      # (absent on objects built without __init__: tests)
            self.monitor.reset()
        self._exec = None           # closed-loop execution context of the current pick (None = plain queue)
        self.grasp_log = []         # one record per executed pick: grasp / transit / release verdicts + recoveries
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
        out = self._act(obs)
        self.monitor.track(out["hand_q_target"])   # every emitted hand target (the monitor needs its history)
        return out

    def _act(self, obs):
        if self.arm_cmd is None:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if self.queue_a is not None and len(self.queue_a):
            if self._exec is not None:
                return self._step_exec(obs, t)
            if self.skill is not None and self.skill.active:
                return self.skill.act(self, obs)
            return self._pop()
        self._exec = None
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
            return self._act(obs)
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
        if self.grasp_log and self.grasp_log[-1]["relook"] is None:
            self.grasp_log[-1]["relook"] = out   # camera cross-check of the finger-load verdicts
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
            r = self.cfg["policy"]["same_object_radius"]
            self.planner.hint_shift = sum(c for q, c in self.attempts if np.linalg.norm(d.top[:2] - q) < r)
            p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, obstacles=others)
            p.det = d
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
        self._begin_exec(plan)
        if self._exec is None:       # open loop: the release is assumed; closed loop counts it when it happened
            self._count_drop(plan)
        self.state = "executing"
        self._note(t, f"execute {plan.summary()}")
        return self._pop()

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)

    def _pop(self, k=None):
        k = int(self.cfg["motion"]["chunk"]) if k is None else max(1, int(k))
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

    def _hold_n(self, n):
        """n ticks of the current target (a pause inside a plan; unlike _hold it is never a `done`)."""
        return {"arm_q_target": np.repeat(self.arm_cmd[None], n, 0), "hand_q_target": np.repeat(self.hand_cmd[None], n, 0),
                "done": False}

    # ------------------------------------------------------------------ closed-loop execution (config `verify`)
    # A plan is still computed as one validated trajectory, but it is no longer played blind:
    #   close -> lift -> VERIFY (pause at the top of the lift until the finger-load monitor says held / empty)
    #         -> carry, MONITORED every `verify.chunk` ticks (object lost = stop, never an empty-handed drop-off)
    #         -> release -> RELEASE CHECK in the `open` posture above the bin (+ recovery if the object is still
    #            hooked in the fingers) -> retreat -> home -> camera re-look (_check_last_place) as the second,
    #            independent check of what really happened.
    # Every recovery path is made of ticks of the validated plan played backwards (plus short joint moves) and is
    # dry-run through the safety filter before it is used.
    def _begin_exec(self, plan):
        ph = {n: (s, e) for n, s, e in plan.phases}
        self._exec = None
        if not self.monitor.enabled or not all(k in ph for k in ("to_pregrasp", "descend", "close", "lift", "open")):
            return
        det = getattr(plan, "det", None)
        rec = {"color": plan.color, "shape": getattr(det, "shape", None), "side": plan.side, "bin": plan.target_bin,
               "grasp": None, "transit": None, "release": None, "recoveries": 0, "relook": None}
        self.grasp_log.append(rec)
        self._exec = {"plan": plan, "side": plan.side, "ph": ph, "stage": "grasp", "ins_a": None, "ins_h": None,
                      "arm0": self.arm_cmd.copy(), "waited": 0, "recover": 0, "rec": rec}
        self.monitor.restart()

    def _count_drop(self, plan):
        """Objects released per bin (planner grasp.drop_spread: the next object lands elsewhere in the bin)."""
        if plan.target_bin in self.scene.bin_names:
            self.planner.drop_count[plan.target_bin] = self.planner.drop_count.get(plan.target_bin, 0) + 1

    def _vlog(self, t, msg):
        self.events.append((round(t, 2), "verify: " + msg))
        self.log(f"[verify t={t:6.2f}] {msg}")

    def _insert(self, hand_targets, holds):
        """Queue hand-only moves (arm fixed) ahead of the rest of the plan: ramp to each posture, then hold it."""
        ex, hv = self._exec, self.cfg["motion"]["hand_max_vel"]
        cur, rows = self.hand_cmd.copy(), []
        for to, hold_s in zip(hand_targets, holds):
            r = ramp(cur, to, hv, 1.0 / HZ)
            rows += [r, np.repeat(np.asarray(to, float)[None], int(round(hold_s * HZ)), 0)]
            cur = np.asarray(to, float).copy()
        h = np.vstack(rows)
        ex["ins_a"], ex["ins_h"] = np.repeat(self.arm_cmd[None], len(h), 0), h

    def _pop_ins(self):
        ex, k = self._exec, max(1, int(self.cfg["verify"].get("chunk", 5)))
        a, h = ex["ins_a"][:k], ex["ins_h"][:k]
        ex["ins_a"], ex["ins_h"] = ex["ins_a"][k:], ex["ins_h"][k:]
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _step_exec(self, obs, t):
        ex, v = self._exec, self.cfg["verify"]
        if ex["ins_a"] is not None and len(ex["ins_a"]):
            return self._pop_ins()
        plan, side, ph, rec = ex["plan"], ex["side"], ex["ph"], ex["rec"]
        sl = side_slice(side)
        k = len(plan.arm) - len(self.queue_a)          # plan ticks already handed out
        vchunk = max(1, int(v.get("chunk", 5)))
        skill_on = self.skill is not None and self.skill.active
        if ex["stage"] == "grasp":
            if k < ph["lift"][1]:
                if skill_on:
                    return self.skill.act(self, obs)
                if k < ph["close"][0]:
                    return self._pop(min(int(self.cfg["motion"]["chunk"]), ph["close"][0] - k))
                return self._pop(min(vchunk, ph["lift"][1] - k))
            ex["stage"] = "verify"
            self.monitor.restart()
        if ex["stage"] == "verify":       # top of the lift: nothing moves on until the grasp is confirmed
            verdict = self.monitor.sample(obs, self.hand_cmd, side)
            th, fi = self.monitor.last_load
            if verdict == "held":
                rec["grasp"] = "confirmed"
                self._vlog(t, f"grasp confirmed ({plan.color} {side}): finger load thumb {th:.3f} fingers {fi:.3f} rad")
                ex["stage"] = "carry"
                self.monitor.restart()
            else:
                if verdict == "empty":
                    return self._abort("grasp_failed", t, f"hand empty after the lift (load thumb {th:.3f} fingers {fi:.3f})")
                if verdict == "unsure":
                    if ex["waited"] >= int(round(float(v.get("max_wait_s", 1.0)) * HZ)):
                        return self._abort("grasp_unconfirmed", t, f"grasp not confirmed within {v.get('max_wait_s', 1.0)} s "
                                           f"(load thumb {th:.3f} fingers {fi:.3f})")
                    ex["waited"] += vchunk
                return self._hold_n(vchunk)
        if ex["stage"] == "carry":
            if k < ph["open"][0]:
                verdict = self.monitor.sample(obs, self.hand_cmd, side)
                if verdict == "empty":
                    th, fi = self.monitor.last_load
                    return self._abort("lost_in_transit", t, f"object lost during the carry at plan tick {k} "
                                       f"(load thumb {th:.3f} fingers {fi:.3f})")
                if skill_on:
                    return self.skill.act(self, obs)
                return self._pop(min(vchunk, ph["open"][0] - k))
            rec["transit"] = "ok"
            ex["stage"] = "release"
        if ex["stage"] == "release":
            if k < ph["open"][1]:
                if skill_on:      # the RL hand residual is taken out by the skill itself (rate-limited), not by a jump
                    return self.skill.act(self, obs)
                return self._pop(min(vchunk, ph["open"][1] - k))
            if self.skill is not None:
                self.skill.active = False
            rc = v.get("release") or {}
            if not rc.get("check", True):
                ex["stage"] = "finish"
            else:
                hand_open = self.hand_cmd.copy(); hand_open[sl] = self.post["open"][sl]
                self._insert([hand_open], [float(rc.get("settle_s", 0.5))])
                ex["stage"] = "release_check"
                return self._pop_ins()
        if ex["stage"] == "release_check":
            rc = v.get("release") or {}
            r = self.monitor.release_reading(obs, self.hand_cmd, side)
            if r == "wait":
                return self._hold_n(vchunk)
            th, fi = self.monitor.last_load
            if r == "stuck" and ex["recover"] < int(rc.get("recover_attempts", 2)):
                ex["recover"] += 1
                rec["recoveries"] = ex["recover"]
                self._vlog(t, f"release NOT clean ({plan.color} {side}): load {th + fi:.3f} rad in the open posture -> "
                              f"recovery {ex['recover']}")
                seq = []
                for name in rc.get("recover_postures", ["scissor_a", "scissor_b", "release"]):
                    h = self.hand_cmd.copy(); h[sl] = self.post[name][sl]
                    seq.append(h)
                hand_open = self.hand_cmd.copy(); hand_open[sl] = self.post["open"][sl]
                hold = float(rc.get("recover_hold_s", 0.3))
                self._insert(seq + [hand_open], [hold] * len(seq) + [float(rc.get("settle_s", 0.5))])
                return self._pop_ins()
            rec["release"] = "failed" if r == "stuck" else ("clean" if ex["recover"] == 0 else "recovered")
            if rec["release"] != "failed":
                self._count_drop(plan)
            self._vlog(t, f"release {rec['release']} ({plan.color} {side}): load {th + fi:.3f} rad, recoveries {ex['recover']}")
            self.queue_h[:, sl] = self.post["open"][sl]     # the hand is in `open` already: no second ramp on the way home
            ex["stage"] = "finish"
        return self._pop()          # finish: retreat + home as planned

    def _abort(self, kind, t, why):
        """Stop the pick here and go back to the start pose along validated ticks of the plan (played backwards).
        grasp_failed (hand empty)        : open, back to the pregrasp pose, home.
        grasp_unconfirmed (maybe holding): back DOWN the lift, open at the grasp pose (puts a half-held object back
                                           where it was instead of dropping it from carry height), up, home.
        lost_in_transit                  : open, back along the carry to the top of the lift, home.
        The next look (camera) then says where the object is; the normal loop retries it (policy.max_attempts)."""
        ex = self._exec
        plan, side, ph, rec = ex["plan"], ex["side"], ex["ph"], ex["rec"]
        sl, mo, dt = side_slice(side), self.cfg["motion"], 1.0 / HZ
        if self.skill is not None:
            self.skill.active = False
        k = len(plan.arm) - len(self.queue_a)
        s_des, s_close, (s_lift, e_lift) = ph["descend"][0], ph["close"][0], ph["lift"]
        cur_a, cur_h = self.arm_cmd.copy(), self.hand_cmd.copy()
        hand_open = cur_h.copy(); hand_open[sl] = self.post["open"][sl]
        back_home = [plan.arm[s_des - 1]] + list(plan.arm[:s_des][::-1]) + [ex["arm0"]]   # pregrasp pose -> start pose

        def path(a_from, Qs):
            Q = [np.asarray(a_from, float)]
            for q in Qs:                                       # drop repeated poses (holds in the plan)
                if np.abs(q - Q[-1]).max() > 1e-9:
                    Q.append(np.asarray(q, float))
            return time_scale(Q, mo["arm_max_vel"], mo["arm_max_acc"], dt)

        def with_hand(a, h_from, h_to):                        # hand ramps at the start of an arm move
            hr = ramp(h_from, h_to, mo["hand_max_vel"], dt)
            n = max(len(a), len(hr))
            a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], n - len(a), 0)]) if len(a) < n else a
            h = np.repeat(np.asarray(h_to, float)[None], n, 0)
            h[:len(hr)] = hr
            return a, h

        if kind == "grasp_unconfirmed":
            a1 = path(cur_a, plan.arm[s_lift - 1:e_lift][::-1])               # down the lift, hand as it is
            h1 = np.repeat(cur_h[None], len(a1), 0)
            hr = ramp(cur_h, hand_open, mo["hand_max_vel"], dt)                # open at the grasp pose + short hold
            h2 = np.vstack([hr, np.repeat(hand_open[None], int(round(0.2 * HZ)), 0)])
            a2 = np.repeat((a1[-1] if len(a1) else cur_a)[None], len(h2), 0)
            a3 = path(a2[-1], list(plan.arm[:s_close][::-1]) + [ex["arm0"]])  # up the descend path, then home
            h3 = np.repeat(hand_open[None], len(a3), 0)
            A, Hd = np.vstack([a1, a2, a3]), np.vstack([h1, h2, h3])
        else:
            carry_back = list(plan.arm[e_lift - 1:k][::-1]) if kind == "lost_in_transit" else []
            A, Hd = with_hand(path(cur_a, carry_back + back_home), cur_h, hand_open)
        clips = self.planner.dry_run_safety(cur_a, cur_h, A, Hd)
        if clips:   # should not happen (plan ticks reversed); fall back to the plain joint move used at episode start
            self._vlog(t, f"recovery path had {clips} predicted safety clips: joint move home instead")
            A, Hd = self.planner.go_to(cur_a, cur_h, ex["arm0"], hand_open)
        rec["grasp" if kind.startswith("grasp") else "transit"] = kind
        self._vlog(t, f"ABORT {kind} ({plan.color} {side}): {why} -> recover ({len(A) / HZ:.1f} s), no transport / release")
        self._enqueue(A, Hd)
        self._exec = None
        return self._pop()

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
