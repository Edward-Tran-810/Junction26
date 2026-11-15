"""Closed-loop execution of a grasp.mode=topdown pick (policy._step_exec delegates here while stage == "grasp").

States (logged as "[TD] <state>" events with the sim time):
  PRE-GRASP        to_pregrasp + fast approach (fingers in td_pre, palm above the object)
  DESCENDING       guarded descent (plan phase "descend", slow): every `guard.chunk` ticks the PhysX contact stream is read
  CONTACT DETECTED the PALM touched the object (>= guard.object_n for guard.confirm_msgs consecutive 100 Hz PhysX
                   messages) -> descent stops at once -> closing. trigger = "palm_contact" (VERIFIED palm contact).
  EARLY CONTACT    a thumb / finger link touched the object before the palm (object taller than estimated) -> descent
                   stops at once (never pushed through) -> closing as a FALLBACK: trigger = "fallback_link_contact".
  COLLISION        a hand link touched something else (table / bin / lid, >= guard.other_n) or any force > max_force_n
                   -> descent stops at once; the pick is aborted (no closing on a wrong contact)
  NEAR-CONTACT     the descent reached the planned near-contact height without contact (palm pad `gap` above the top;
                   TRUE palm contact is geometrically infeasible with a three-finger wrap, STATE.md) -> closing as a
                   FALLBACK: trigger = "fallback_height". Only "palm_contact" counts as palm-contact detection.
  CLOSING          fast ramp to approach_frac of the contact posture, then LEAD-LIMITED closure towards td_goal: per joint
                   command = measured + clip(goal - measured, +-L), L = torque_frac * tau_max / kp (holding torque within
                   the URDF effort limits, never "max effort"). Groups that touch the object switch to lead mode at once.
  CLOSED           thumb + (index or middle) in contact with the SAME object for settle_s and joints steady (or, without a
                   contact stream, joints steady); the hold command is then FROZEN (constant torque, monitor can settle)
  -> the policy's normal flow: small verification lift (plan "lift", 25 mm) -> GRASP VERIFIED (finger load + contacts)
     -> LIFTING (lift2) -> carry (monitored) -> release (hold released, hand ramps from the actual command) -> check.

Contacts are the sim's PhysX normal forces (contacts.py); without the stream the guard falls back to the near-contact
height trigger, and every pick record says which trigger fired.
"""
from __future__ import annotations

import numpy as np

from .contacts import group_forces
from .grasp_monitor import side_slice

HZ = 30.0


class TopDownExec:
    def __init__(self, pol, plan):
        self.pol, self.plan = pol, plan
        cfg = pol.cfg["grasp"]["topdown"]
        self.g, self.sq = cfg.get("guard") or {}, cfg.get("squeeze") or {}
        self.side = plan.side
        self.sl = side_slice(plan.side)
        post = pol.post
        self.pre = post[cfg["preshape_posture"]][self.sl].copy()
        self.contact_q = post[cfg["closed_posture"]][self.sl].copy()
        self.goal = post[cfg["goal_posture"]][self.sl].copy()
        tau = np.asarray(self.sq.get("tau_max", [2.45] + [1.4] * 6), float)
        self.L = float(self.sq.get("torque_frac", 0.85)) * tau / float(self.sq.get("kp", 100.0))   # rad lead per joint
        self.approach_q = self.pre + float(self.sq.get("approach_frac", 0.85)) * (self.contact_q - self.pre)
        # per hand slice: thumb 0-2; left: middle 3-4, index 5-6; right: index 3-4, middle 5-6 (DDS order)
        self.jg = {"thumb": [0, 1, 2], "middle": [3, 4] if plan.side == "left" else [5, 6],
                   "index": [5, 6] if plan.side == "left" else [3, 4]}
        # contacts_for_control False (default, deployable): PhysX contacts are LOGGED for diagnosis / validation only;
        # every control decision uses proprioception (finger tracking load, joint motion) and the planned height.
        self.use_contacts = bool(self.g.get("contacts_for_control", False))
        self.events_c = []          # diagnostic contact events (link, partner, force, sim t)
        self.state = "PRE-GRASP"
        self.closing = None
        self.done_close = False
        self.hold = None            # frozen hold command (7,) once CLOSED
        self.q_ref = None
        self.skip_to = None         # plan tick index to fast-forward to (dropped descent remainder / lift start)
        self.rec = {"trigger": None, "palm_contact": False, "contact_link": None, "contact_obj": None, "contact_force_n": None,
                    "t_descend": None, "t_trigger": None, "t_close_start": None, "t_closed": None, "close_s": None,
                    "groups_at_close": None, "verified_contacts": None, "regrips": 0, "stream": None,
                    "torque_frac": float(self.sq.get("torque_frac", 0.85)), "hold_torque_nm": None, "target_obj": None}
        plan_rec = pol._exec["rec"]
        plan_rec["topdown"] = self.rec

    # ------------------------------------------------------------------ measurement only (sim, AALTO_TD_DIAG=1)
    def diag(self, obs, t, label, every=None):
        """Ground-truth object pose RELATIVE to the hand's grasp point (FK of the MEASURED arm joints), from the sim's
        scene service. MEASUREMENT ONLY (slip / rotation / drop metrics): never used for control."""
        sc = getattr(self.pol, "diag_scene", None)
        if sc is None:
            return
        if every is not None and t - getattr(self, "_t_diag", -1e9) < every:
            return
        self._t_diag = t
        try:
            objs = [o for o in sc.state()["objects"] if o["active"]]
        except Exception as e:                     # never break the controller over a diagnostic
            self.rec.setdefault("diag_errors", []).append(repr(e)[:80]); return
        kin = self.pol.kin                         # the TOP-DOWN grasp point (object centre target), not the config's
        td = self.pol.cfg["grasp"]["topdown"]      # legacy point_in_palm that kin.offset holds outside planning
        sh = getattr(getattr(self.plan, "det", None), "shape", None)
        H = self.pol.scene.object_height(sh) if sh else self.pol.planner.height
        pt = np.array([float(td["palm_x"]), -(float(td["pad_y"]) + float(td["gap"]) + H / 2), 0.0])
        saved = kin.offset
        kin.offset = {"left": pt, "right": pt * np.array([1.0, -1.0, 1.0])}
        try:
            g = kin.grasp_pose(self.side, np.asarray(obs["arm_q"], float))[0]
        finally:
            kin.offset = saved
        name = self.rec.get("target_obj")
        ob = next((o for o in objs if o["name"] == name), None) or min(objs, key=lambda o: np.linalg.norm(np.subtract(o["pos"], g)))
        rel = np.subtract(ob["pos"], g)
        if label in ("trigger", "closed"):
            self.rec[f"clearance_model_{label}"] = self._model_clearance(obs)
        self.rec.setdefault("diag", []).append({"t": round(t, 3), "label": label, "obj": ob["name"], "rel_mm": np.round(rel * 1e3, 1).tolist(),
                                                 "obj_pos": ob["pos"], "quat_xyzw": ob["quat_xyzw"], "speed": ob["speed"],
                                                 "grasp_pt": np.round(g, 4).tolist()})

    def _model_clearance(self, obs):
        """DIAGNOSTIC: lowest point of each Dex3 link above the counter (mm), from the MEASURED arm + hand joints and the
        official Dex3 collision meshes (experiments/topdown_contact). A model estimate (the sim thumb hull / joint stop
        can differ by a few mm); the physical truth is the PhysX contact report."""
        try:
            from aaltonomous.experiments.topdown_contact.posture_check import Hand
            H = getattr(self.pol, "_diag_hand", None) or Hand(400)
            self.pol._diag_hand = H
            kin, zc = self.pol.kin, self.pol.scene.counter_z
            kin._fk(np.asarray(obs["arm_q"], float)); M = kin.data.oMf[kin.palm[self.side]]
            hq = np.asarray(obs["hand_q"], float)[self.sl]
            if self.side == "right":
                hq = -np.array([hq[0], hq[1], hq[2], hq[5], hq[6], hq[3], hq[4]])
            P = H.posed(hq, [k for k in H.pts if k != "palm"])
            names = {"t0": "thumb_0", "t1": "thumb_1", "t2": "thumb_2", "m0": "middle_0", "m1": "middle_1", "i0": "index_0", "i1": "index_1"}
            out = {}
            for k, v in P.items():
                if self.side == "right":
                    v = v * np.array([1.0, -1.0, 1.0])
                W = v @ M.rotation.T + M.translation
                out[f"{self.side}_hand_{names[k]}_link"] = round(float(W[:, 2].min() - zc) * 1e3, 1)
            low = min(out, key=out.get)
            return {"lowest_link": low, "lowest_mm": out[low], "thumb_mm": min(out[f"{self.side}_hand_thumb_{i}_link"] for i in range(3)), "per_link_mm": out}
        except Exception as e:
            return {"error": repr(e)[:120]}

    def dump(self, t, outcome):
        """Append this pick's record (one JSON line) to $AALTO_TD_LOG, once."""
        import json, os
        path = os.environ.get("AALTO_TD_LOG")
        if not path or getattr(self, "_dumped", False):
            return
        self._dumped = True
        r = dict(self.rec, outcome=outcome, t_end=round(t, 3), side=self.side, color=self.plan.color,
                 shape=getattr(getattr(self.plan, "det", None), "shape", None), plan_grasp=self.plan.grasp,
                 states=[e for e in self.pol.events if "[TD]" in e[1]][-30:])
        with open(path, "a") as f:
            f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")

    # ------------------------------------------------------------------ helpers
    def log(self, t, state, msg=""):
        self.state = state
        self.pol.events.append((round(t, 2), f"[TD] {state} {msg}".rstrip()))
        self.pol.log(f"[TD t={t:6.2f}] {state} {msg}")

    def stream(self, control=False):
        """The contact stream. control=True: only when contacts may drive decisions (guard.contacts_for_control)."""
        if control and not self.use_contacts:
            return None
        s = self.pol.contacts
        return s if (s is not None and s.available) else None

    def forces(self, k=None, control=False):
        s = self.stream(control)
        if s is None:
            return []
        k = k or max(1, int(self.g.get("confirm_msgs", 3)))
        return [group_forces(m, self.side) for m in s.recent(k)]

    def _rate_limited(self, cur, want, n):
        """n ticks from cur to want, the command change per tick capped at motion.hand_max_vel (no safety clips)."""
        step = float(self.pol.cfg["motion"]["hand_max_vel"]) / HZ
        out, c = [], np.asarray(cur, float).copy()
        for _ in range(n):
            c = c + np.clip(want - c, -step, step)
            out.append(c.copy())
        return np.array(out)

    def _emit_hand(self, hand7_rows):
        """Arm held, grasping hand following hand7_rows (n, 7)."""
        pol = self.pol
        n = len(hand7_rows)
        a = np.repeat(pol.arm_cmd[None], n, 0)
        h = np.repeat(pol.hand_cmd[None], n, 0)
        h[:, self.sl] = hand7_rows
        pol.hand_cmd = h[-1].copy()
        return {"arm_q_target": a, "hand_q_target": h, "done": False}

    def _consume_to(self, idx):
        """Drop plan ticks up to plan index idx without emitting them (the policy's k follows the queue length)."""
        pol = self.pol
        k = len(self.plan.arm) - len(pol.queue_a)
        n = max(0, idx - k)
        if n:
            pol.queue_a, pol.queue_h = pol.queue_a[n:], pol.queue_h[n:]

    # ------------------------------------------------------------------ main
    def step(self, obs, t, k, ph):
        """Action for this tick while the pick is in the grasp stage, or None to let the policy pop the plan."""
        pol = self.pol
        chunk = max(1, int(self.g.get("chunk", 2)))
        if self.skip_to is not None and k < self.skip_to:
            self._consume_to(self.skip_to)
            k = self.skip_to
        self.skip_to = None
        d0, d1 = ph["descend"]
        c0, c1 = ph["close"]
        if self.rec["stream"] is None:
            self.rec["stream"] = self.stream() is not None
        # ---- PRE-GRASP / fast approach: watch for unexpected contacts (abort), pop the plan
        if k < d0:
            if self.state != "PRE-GRASP":
                self.log(t, "PRE-GRASP")
            bad = self._collision()
            if bad:
                return pol._abort("grasp_failed", t, f"[TD] unexpected contact before the guarded descent: {bad}")
            return pol._pop(min(int(pol.cfg["motion"]["chunk"]), d0 - k))
        # ---- DESCENDING (guarded)
        if k < c0 and not self.done_close and self.closing is None:
            if self.state != "DESCENDING":
                self.rec["t_descend"] = round(t, 3)
                self.log(t, "DESCENDING", f"guarded, contact stream {'ON' if self.stream() else 'OFF (height trigger only)'}")
            self._log_contacts(t)
            pro = self._proprio_block(obs)
            if pro:
                h = self._heights(obs)
                self.rec.update(trigger="proprio_block", t_trigger=round(t, 3), collision_heights=h, stop_tick=[k, c0],
                                proprio=pro)
                self.log(t, "BLOCKED", f"{pro}; {h}: descent stopped")
                return pol._abort("grasp_failed", t, f"[TD] hand blocked during the guarded descent (proprioceptive): {pro}")
            hit = self._object_contact()
            if hit:
                link, obj, f = hit
                palm = link == "palm"
                self.rec.update(trigger="palm_contact" if palm else "fallback_link_contact", palm_contact=palm, contact_link=link,
                                contact_obj=obj, contact_force_n=round(f, 3), t_trigger=round(t, 3), stop_tick=[k, c0])
                self.rec["target_obj"] = obj
                if palm:
                    self.log(t, "CONTACT DETECTED", f"PALM vs {obj} {f:.2f} N (verified, {self.g.get('confirm_msgs', 3)} msgs) "
                                                    f"at plan tick {k}/{c0}: descent stopped")
                else:
                    self.log(t, "EARLY CONTACT", f"{link} (not the palm) vs {obj} {f:.2f} N at plan tick {k}/{c0}: descent "
                                                 f"stopped; closing as FALLBACK (no palm contact)")
                self._stop_descent(k, ph)
                return self._start_close(obs, t)
            bad = self._collision()
            if bad:
                h = self._heights(obs)
                self.rec.update(trigger="collision", t_trigger=round(t, 3), collision_heights=h, stop_tick=[k, c0])
                self.log(t, "COLLISION", f"{bad}; {h}")
                return pol._abort("grasp_failed", t, f"[TD] collision during the guarded descent: {bad}")
            return pol._pop(min(chunk, c0 - k))
        # ---- trigger at the near-contact height
        if self.closing is None and not self.done_close:
            self.rec.update(trigger="fallback_height", t_trigger=round(t, 3))
            self.log(t, "NEAR-CONTACT", "FALLBACK: planned height reached, no palm contact (palm pad ~gap above the top)")
            return self._start_close(obs, t)
        # ---- CLOSING
        if self.closing is not None:
            return self._close_step(obs, t, k, ph, chunk)
        return None        # closed: lift etc. from the plan (hold applied by grip())

    # ------------------------------------------------------------------ descent guard
    def _log_contacts(self, t):
        """DIAGNOSTIC: record every new hand contact (exact link, partner, force) from the PhysX stream."""
        F = self.forces(1)
        if not F:
            return
        for grp in ("palm", "thumb", "index", "middle"):
            f = F[-1][grp]
            for kind, key, lk in (("object", "obj", "obj_link"), ("table/bin/lid", "other", "other_link")):
                if f[key] >= 0.05 and (grp, kind) not in {(e["group"], e["partner"]) for e in self.events_c}:
                    e = {"t": round(t, 3), "state": self.state, "group": grp, "link": f[lk], "partner": kind,
                         "force_n": round(f[key], 3), "objs": f["objs"] if key == "obj" else None}
                    self.events_c.append(e)
                    self.rec.setdefault("contact_events", []).append(e)
                    self.log(t, "CONTACT EVENT (diagnostic)", f"{f[lk]} vs {kind} {f[key]:.2f} N")

    def _proprio_block(self, obs):
        """Deployable descent guard: a hand joint held off its (constant) command by > guard.proprio_block_rad means a
        finger / the thumb is resting on something (the arm keeps descending, the finger cannot)."""
        thr = float(self.g.get("proprio_block_rad", 0.03))
        if self.pol.monitor.steady.get(self.side, 0) < 10:
            return None
        err = np.asarray(obs["hand_q"], float)[self.sl] - self.pol.hand_cmd[self.sl]
        if getattr(self, "err0", None) is None:        # baseline at the start of the guarded descent: a joint that cannot
            self.err0 = err.copy()                     # reach its command (sim thumb_2 stop) is a constant offset, not a block
            self.rec["hand_offset_at_descent"] = np.round(err, 3).tolist()
            return None
        e = np.abs(err - self.err0)
        if e.max() > thr:
            names = ["thumb_0", "thumb_1", "thumb_2"] + (["middle_0", "middle_1", "index_0", "index_1"] if self.side == "left" else ["index_0", "index_1", "middle_0", "middle_1"])
            j = int(e.argmax())
            return f"{self.side}_hand_{names[j]}_joint off its command by {e[j]:.3f} rad"
        return None

    def _object_contact(self):
        """(link group, object, force) when a hand group touched an object in all of the last confirm_msgs messages
        (only with guard.contacts_for_control; otherwise None: the height / proprioceptive triggers decide)."""
        F = self.forces(control=True)
        if not F:
            return None
        thr = float(self.g.get("object_n", 0.3))
        for grp in ("palm", "thumb", "index", "middle"):
            if all(f[grp]["obj"] >= thr for f in F):
                objs = F[-1][grp]["objs"]
                name = max(objs, key=objs.get)
                return grp, name, objs[name]
        return None

    def _collision(self):
        F = self.forces(control=True)
        if not F:
            return None
        thr, fmax = float(self.g.get("other_n", 1.0)), float(self.g.get("max_force_n", 15.0))
        for grp in ("palm", "thumb", "index", "middle"):
            if all(f[grp]["other"] >= thr for f in F):
                return f"{grp} touched the table / a bin / the lid ({F[-1][grp]['other']:.2f} N)"
            if all(f[grp]["net"] >= fmax for f in F):
                return f"{grp} force {F[-1][grp]['net']:.1f} N > {fmax} N"
        return None

    def _heights(self, obs):
        """Palm pad height above the counter (mm): MEASURED arm joints vs the COMMANDED arm target, and the planned
        height at the end of the descent (diagnoses arm sag / geometry error when something touches the table)."""
        kin, zc = self.pol.kin, self.pol.scene.counter_z
        td = self.pol.cfg["grasp"]["topdown"]
        def pad(q):
            kin._fk(np.asarray(q, float)); M = kin.data.oMf[kin.palm[self.side]]
            n = M.rotation @ np.array([0.0, -1.0 if self.side == "left" else 1.0, 0.0])    # palm normal (down)
            return float((M.translation + float(td["pad_y"]) * n)[2] - zc) * 1e3
        e = dict(self.pol._exec["ph"])["descend"][1] - 1
        return {"pad_meas_mm": round(pad(obs["arm_q"]), 1), "pad_cmd_mm": round(pad(self.pol.arm_cmd), 1),
                "pad_plan_end_mm": round(pad(self.plan.arm[e]), 1),
                "hand_meas": np.round(np.asarray(obs["hand_q"], float)[self.sl], 3).tolist(),
                "hand_cmd": np.round(self.pol.hand_cmd[self.sl], 3).tolist()}

    def _stop_descent(self, k, ph):
        """Drop the rest of the descent and hold the arm where it is; the lift later starts at the plan tick whose grasp
        point is first at or above the current height (no jump, no push towards the counter)."""
        pol, kin = self.pol, self.pol.kin
        c0, c1 = ph["close"]
        l0, l1 = ph["lift"]
        z_now = kin.grasp_pose(self.side, pol.arm_cmd)[0][2]
        j = l0
        for i in range(l0, l1):
            if kin.grasp_pose(self.side, self.plan.arm[i])[0][2] >= z_now - 5e-4:
                j = i
                break
        else:
            j = l1
        self.lift_from = j
        self._consume_to(c0)

    # ------------------------------------------------------------------ closing
    def _start_close(self, obs, t):
        self.diag(obs, t, "trigger")
        self.closing = {"t0": t, "steady": 0, "q_prev": np.asarray(obs["hand_q"], float)[self.sl].copy(), "lead": set()}
        self.rec["t_close_start"] = round(t, 3)
        self.log(t, "CLOSING", f"lead-limited, L={np.round(self.L, 4).tolist()} rad (= {self.rec['torque_frac']:.2f} x tau_max)")
        return self._emit_hand(np.repeat(self.pol.hand_cmd[self.sl][None], 1, 0))

    def _contact_groups(self, control=False):
        F = self.forces(control=control)
        if not F:
            return set(), {}
        thr = float(self.g.get("object_n", 0.3))
        groups, objs = set(), {}
        for grp in ("thumb", "index", "middle"):
            if all(f[grp]["obj"] >= thr for f in F):
                groups.add(grp)
                objs[grp] = set(F[-1][grp]["objs"])
        return groups, objs

    def _close_step(self, obs, t, k, ph, chunk):
        pol, c = self.pol, self.closing
        q = np.asarray(obs["hand_q"], float)[self.sl]
        cur = pol.hand_cmd[self.sl].copy()
        self._log_contacts(t)
        groups, objs = self._contact_groups(control=True)
        diag_groups, diag_objs = self._contact_groups()          # logged only
        c["diag_groups"] = diag_groups
        c["lead"] |= groups
        want = cur.copy()
        for grp, idx in self.jg.items():
            idx = np.array(idx)
            reached = np.all(np.abs(cur[idx] - self.approach_q[idx]) < 1e-3)
            if grp in c["lead"] or reached or grp in c.get("lead_auto", set()):
                if not grp in c["lead"]:
                    c.setdefault("lead_auto", set()).add(grp)
                want[idx] = q[idx] + np.clip(self.goal[idx] - q[idx], -self.L[idx], self.L[idx])
            else:
                want[idx] = self.approach_q[idx]
        rows = self._rate_limited(cur, want, chunk)
        dq = float(np.abs(q - c["q_prev"]).max())
        c["q_prev"] = q.copy()
        common = None
        if objs:
            sets = [objs[g] for g in objs]
            common = set.intersection(*sets) if sets else None
        need = self.sq.get("need", ["thumb", "finger"])
        have = ("thumb" in groups or "thumb" not in need) and (("index" in groups or "middle" in groups) or "finger" not in need)
        stream = self.stream(control=True) is not None
        all_lead = all(g in c["lead"] or g in c.get("lead_auto", set()) for g in self.jg)
        ok = (have and bool(common)) if stream else all_lead
        c["steady"] = c["steady"] + chunk if (ok and dq < 0.003) else 0
        el = t - c["t0"]
        if c["steady"] >= int(round(float(self.sq.get("settle_s", 0.4)) * HZ)) or el >= float(self.sq.get("max_s", 3.0)):
            if not stream:                      # diagnostic record of what really touched at the end of closing
                groups = diag_groups
                common = set.intersection(*[diag_objs[g] for g in diag_objs]) if diag_objs else None
            return self._finish_close(obs, t, k, ph, groups, common, timeout=c["steady"] < int(round(float(self.sq.get("settle_s", 0.4)) * HZ)))
        return self._emit_hand(rows)

    def _finish_close(self, obs, t, k, ph, groups, common, timeout):
        pol = self.pol
        q = np.asarray(obs["hand_q"], float)[self.sl]
        self.hold = q + np.clip(self.goal - q, -self.L, self.L)
        self.q_ref = q.copy()
        tau = float(self.sq.get("kp", 100.0)) * np.abs(self.hold - q)
        F = self.forces(1)
        self.rec["forces_at_closed_n"] = {g: round(F[-1][g]["obj"], 3) for g in ("palm", "thumb", "index", "middle")} if F else None
        self.rec.update(t_closed=round(t, 3), close_s=round(t - self.closing["t0"], 3), groups_at_close=sorted(groups),
                        hold_torque_nm=np.round(tau, 3).tolist(), target_obj=self.rec["target_obj"] or (sorted(common)[0] if common else None))
        self.log(t, "CLOSED" if not timeout else "CLOSED (timeout)",
                 f"contacts {sorted(groups) or 'none'} on {sorted(common) if common else '-'}; hold torque <= {tau.max():.2f} N m")
        self.closing, self.done_close = None, True
        self.diag(obs, t, "closed")
        c0, c1 = ph["close"]
        self._consume_to(c1)
        if getattr(self, "lift_from", None) is not None:
            self.skip_to = self.lift_from
        rows = self._rate_limited(pol.hand_cmd[self.sl], self.hold, 3)
        return self._emit_hand(rows)

    # ------------------------------------------------------------------ hold during lift / carry
    def grip(self, out, obs, t, prev_cmd):
        """Replace the grasping hand's planned columns with the frozen hold command (rate limited), re-gripping when the
        fingers moved by > 8 mrad towards the goal (= the object moved in the hand: counted as a slip event)."""
        if self.hold is None:
            return out
        pol = self.pol
        q = np.asarray(obs["hand_q"], float)[self.sl]
        moved = float(np.abs(q - self.q_ref).max())
        # re-grip only for a real slip: fingers moved past the reference AND are still blocked (a lead remains), at most
        # max_regrips per pick; an empty hand just closes onto the frozen hold (the monitor then says "empty")
        blocked = float(np.abs(self.hold - q).max()) > 0.5 * float(self.L[1:].min())
        if moved > 0.008 and blocked and self.rec["regrips"] < int(self.sq.get("max_regrips", 3)):
            self.rec["regrips"] += 1
            self.hold = q + np.clip(self.goal - q, -self.L, self.L)
            self.q_ref = q.copy()
            self.log(t, "REGRIP", f"fingers moved {moved:.3f} rad since the hold was set (object moved in the hand?)")
        self.diag(obs, t, "hold", every=1.0)
        h = np.asarray(out["hand_q_target"], float).copy()
        start = (prev_cmd if prev_cmd is not None else pol.hand_cmd)[self.sl]
        h[:, self.sl] = self._rate_limited(start, self.hold, len(h))
        out["hand_q_target"] = h
        pol.hand_cmd = h[-1].copy()
        return out

    def verify_contacts(self):
        """Contacts at verification: always RECORDED (diagnostic); returned as a veto only with contacts_for_control."""
        if self.stream() is None:
            return None
        groups, objs = self._contact_groups()
        ok = "thumb" in groups and ("index" in groups or "middle" in groups)
        self.rec["verified_contacts"] = sorted(groups)
        self.rec["verified_contacts_ok"] = ok
        return ok if self.use_contacts else None

    def release(self, k, ph):
        """Stop holding. The plan's open phase ramps from its planned `closed` columns; rewrite the grasping hand's
        columns of that phase as a rate-limited path from the ACTUAL command to the phase's final posture."""
        pol = self.pol
        self.hold = None
        n = ph["open"][1] - k
        if n > 0 and pol.queue_h is not None and len(pol.queue_h) >= n:
            tgt = pol.queue_h[n - 1, self.sl].copy()
            pol.queue_h[:n, self.sl] = self._rate_limited(pol.hand_cmd[self.sl], tgt, n)
