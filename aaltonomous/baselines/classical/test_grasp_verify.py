"""Regression tests for the four grasp failures seen in Isaac on 2026-11-14 (mock world, no simulator needed).

    1 release fails            object stays in the opened hand            -> release check + scissor recovery
    2 wedged between fingers   (same symptom; cause = grasp geometry)     -> grasp_select never centres a narrow
                                                                             object between index and middle
    3 failed grasp, carries on empty hand goes to the bin and "releases"  -> verify after the lift, abort, retry
    4 dropped during transport object falls, arm carries on               -> monitor during the carry, abort, retry

The mock backend injects each failure (mock_world.RenderMockBackend `faults`) and reports the finger tracking error
a real position-controlled hand shows with an object in it (`contact_err`). Every test also runs the policy with
the fix switched OFF and checks that the failure is then really there (a test that cannot fail proves nothing).

    python -m unittest aaltonomous/baselines/classical/test_grasp_verify.py
"""
import os
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.grasp_monitor import GraspMonitor, finger_load
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy

RL_ACTOR = Path(os.environ.get("AALTO_RL_TEST_ACTOR",
                               "/root/long/aaltonomous-humanoid-rl-grasp/runs/rl/r01_base_s1/export_model_399/policy.pt"))


def run_episode(faults=None, overrides=None, seed=0, rl=False):
    """One mock episode. Returns a dict with the policy, the env and ground-truth counters measured per tick."""
    old = os.environ.pop("AALTO_RL_GRASP", None)
    if rl:
        os.environ["AALTO_RL_GRASP"] = str(RL_ACTOR)
    try:
        env = make_mock_env(seed=seed, faults=faults or {})
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, overrides=overrides)
    finally:
        os.environ.pop("AALTO_RL_GRASP", None)
        if old is not None:
            os.environ["AALTO_RL_GRASP"] = old
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    be, sc = env.backend, env.scene
    boxes = [sc.bin_inner_box(n) for n in sc.bin_names]

    def over_bin(p_pelvis):                      # xy inside a bin's inner footprint (any height)
        pm = sc.pelvis_to_mat(p_pelvis)
        return any(np.all(pm[:2] >= lo[:2]) and np.all(pm[:2] <= hi[:2]) for lo, hi in boxes)
    empty_over_bin = 0          # ticks: a CLOSED hand with NOTHING in it is over a bin (= empty-handed transport)
    open_while_stuck_home = 0   # ticks: an object is still in an opened hand while the arm is back at home
    actions = []
    while True:
        act = pol.act(obs)
        actions.append((np.array(act["arm_q_target"]), np.array(act["hand_q_target"])))
        obs, info = env.step(act)
        for side in ("left", "right"):
            g, _ = be.kin.grasp_pose(side, be.arm_q)
            if be._closure(side) > 0.6 and be.held[side] is None and over_bin(g):
                empty_over_bin += 1
            if be.stuck[side] and np.abs(be.arm_q - pol.home).max() < 0.02:
                open_while_stuck_home += 1
        if info["episode_over"]:
            break
    return {"pol": pol, "env": env, "info": info, "outcomes": be.outcomes(pol.mapping), "faults": be.fault_log,
            "empty_over_bin": empty_over_bin, "stuck_at_home": open_while_stuck_home, "actions": actions,
            "clips": info["safety_summary"]["clipped_steps"]}


OPEN_LOOP = {"verify.enabled": False}


class TestMonitor(unittest.TestCase):
    """Thresholds against the loads recorded in the Aalto Isaac sim (see grasp_monitor.py)."""

    def setUp(self):
        self.m = GraspMonitor(load_config())

    def test_recorded_sim_loads(self):
        self.assertEqual(self.m.classify(0.022, 0.029), "held")      # weakest of 17 held picks
        self.assertEqual(self.m.classify(0.000, 0.006), "empty")     # strongest of 7 empty picks
        self.assertEqual(self.m.classify(0.000, 0.030), "unsure")    # fingers resting on something, thumb free
        self.assertEqual(self.m.classify(0.030, 0.000), "unsure")    # one joint group alone is never a grasp

    def test_needs_settled_posture_and_repeated_readings(self):
        m = self.m
        cmd = np.zeros(14)
        meas = cmd.copy(); meas[1] += 0.03; meas[3] += 0.03          # left thumb + finger loaded
        obs = {"hand_q": meas}
        m.track(np.repeat(cmd[None], 5, 0))
        self.assertEqual(m.sample(obs, cmd, "left"), "wait")         # posture constant for 5 ticks only
        m.track(np.repeat(cmd[None], m.settle_ticks, 0))
        m.restart()
        self.assertEqual(m.sample(obs, cmd, "left"), "unsure")       # first reading: not a verdict yet
        for _ in range(4):
            m.track(np.repeat(cmd[None], 5, 0))
            v = m.sample(obs, cmd, "left")
        self.assertEqual(v, "held")
        self.assertEqual(m.sample({"hand_q": cmd}, cmd, "left"), "unsure")   # one empty reading does not flip it
        moved = cmd.copy(); moved[0] = 0.5
        m.track(moved[None])
        self.assertEqual(m.sample(obs, moved, "left"), "wait")       # posture changed: readings are not trusted
        self.assertEqual(finger_load(meas, cmd, "right"), (0.0, 0.0))


class TestNominal(unittest.TestCase):
    def test_nominal_episode_unchanged_outcome_and_no_false_alarm(self):
        r = run_episode()
        self.assertEqual(r["clips"], 0)
        self.assertTrue(all(o == "correct" for o in r["outcomes"]), r["outcomes"])
        log = r["pol"].grasp_log
        self.assertEqual(len(log), len(r["outcomes"]))               # one pick per object: no false abort
        self.assertTrue(all(g["grasp"] == "confirmed" and g["transit"] == "ok" and g["release"] == "clean"
                            and g["relook"] == "left_counter" for g in log), log)


class TestIssue3FailedGraspNoEmptyTransport(unittest.TestCase):
    def test_open_loop_reproduces_the_failure(self):
        r = run_episode({"miss": [0]}, OPEN_LOOP)
        self.assertGreater(r["empty_over_bin"], 0)                   # closed empty hand travelled to the bin

    def test_abort_retry_and_never_over_a_bin_empty(self):
        r = run_episode({"miss": [0]})
        log = r["pol"].grasp_log
        self.assertEqual(r["clips"], 0)
        self.assertEqual(log[0]["grasp"], "grasp_failed")
        self.assertIsNone(log[0]["release"])                         # no release action for a grasp that failed
        self.assertEqual(log[0]["relook"], "still_on_counter")       # camera agrees with the fingers
        self.assertEqual(r["empty_over_bin"], 0)
        self.assertEqual(log[1]["color"], log[0]["color"])           # retried straight away ...
        self.assertTrue(all(o == "correct" for o in r["outcomes"]), r["outcomes"])   # ... and sorted

    def test_every_grasp_missing_gives_up_safely(self):
        r = run_episode({"miss": list(range(40))})
        self.assertEqual(r["clips"], 0)
        self.assertEqual(r["empty_over_bin"], 0)
        self.assertIn(r["info"]["end_reason"], ("done", "time_limit"))
        self.assertTrue(all(g["grasp"] == "grasp_failed" for g in r["pol"].grasp_log))
        self.assertNotIn("wrong", r["outcomes"])


class TestIssue4DropDuringTransport(unittest.TestCase):
    def test_open_loop_reproduces_the_failure(self):
        r = run_episode({"drop": [1]}, OPEN_LOOP)
        self.assertGreater(r["empty_over_bin"], 0)

    def test_loss_detected_transport_stopped_object_recovered(self):
        r = run_episode({"drop": [1]})
        log = r["pol"].grasp_log
        self.assertEqual(r["clips"], 0)
        self.assertEqual(log[1]["grasp"], "confirmed")               # it WAS held: this is not a failed grasp
        self.assertEqual(log[1]["transit"], "lost_in_transit")
        self.assertIsNone(log[1]["release"])
        self.assertEqual(log[1]["relook"], "dropped_elsewhere")
        self.assertEqual(r["empty_over_bin"], 0)
        self.assertTrue(all(o == "correct" for o in r["outcomes"]), r["outcomes"])


class TestIssue1And2ReleaseAndWedge(unittest.TestCase):
    def test_without_recovery_the_object_comes_home_in_the_hand(self):
        r = run_episode({"stuck": [0]}, {"verify.release.recover_attempts": 0})
        self.assertEqual(r["pol"].grasp_log[0]["release"], "failed")     # detected and REPORTED, not assumed fine
        self.assertGreater(r["stuck_at_home"], 0)

    def test_open_loop_does_not_even_notice(self):
        r = run_episode({"stuck": [0]}, OPEN_LOOP)
        self.assertGreater(r["stuck_at_home"], 0)
        self.assertEqual(r["pol"].grasp_log, [])

    def test_release_verified_and_recovered_above_the_bin(self):
        r = run_episode({"stuck": [0]})
        log = r["pol"].grasp_log
        self.assertEqual(r["clips"], 0)
        self.assertEqual(log[0]["release"], "recovered")
        self.assertGreaterEqual(log[0]["recoveries"], 1)
        self.assertEqual(r["stuck_at_home"], 0)
        self.assertTrue(all(o == "correct" for o in r["outcomes"]), r["outcomes"])   # freed INTO its bin

    def test_recovery_postures_stay_inside_the_filter_limits(self):
        from aaltonomous.baselines.classical.config import hand_postures
        from aaltonomous.interface.safety import JointSpec, SafetyConfig
        spec, sc = JointSpec(), SafetyConfig()
        post = hand_postures(load_config(), spec)
        for name in ("scissor_a", "scissor_b", "release", "open"):
            self.assertTrue(np.all(post[name] >= spec.hand_lower + sc.hand_pos_margin - 1e-9), name)
            self.assertTrue(np.all(post[name] <= spec.hand_upper - sc.hand_pos_margin + 1e-9), name)


@unittest.skipUnless(RL_ACTOR.exists(), f"exported RL actor not found: {RL_ACTOR}")
class TestRLIntegration(unittest.TestCase):
    def test_rl_off_is_exactly_the_classical_policy(self):
        a, b = run_episode(), run_episode()
        self.assertIsNone(a["pol"].skill)
        self.assertEqual(len(a["actions"]), len(b["actions"]))
        self.assertTrue(all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                            for x, y in zip(a["actions"], b["actions"])))

    def test_rl_on_contributes_and_stays_inside_the_safety_filter(self):
        c, r = run_episode(), run_episode(rl=True)
        self.assertIsNotNone(r["pol"].skill)
        self.assertGreater(len(r["pol"].skill.times), 100)               # the actor ran
        A = np.vstack([x[0] for x in r["actions"]]); B = np.vstack([x[0] for x in c["actions"]])
        n = min(len(A), len(B))
        self.assertGreater(np.abs(A[:n] - B[:n]).max(), 1e-3)            # and changed the commands
        self.assertEqual(r["clips"], 0)
        self.assertTrue(all(g["grasp"] == "confirmed" for g in r["pol"].grasp_log), r["pol"].grasp_log)

    def test_rl_arm_residual_fades_without_a_step(self):
        """After the lift the arm residual is ramped out: no command step larger than the plan's own steps + ramp."""
        def jerk(fade):
            old = os.environ.get("AALTO_RL_LIMITS")
            os.environ["AALTO_RL_LIMITS"] = '{"fade_ticks": %d}' % fade
            try:
                r = run_episode(rl=True)
            finally:
                os.environ.pop("AALTO_RL_LIMITS")
                if old is not None:
                    os.environ["AALTO_RL_LIMITS"] = old
            A = np.vstack([x[0] for x in r["actions"]])
            d2 = np.abs(np.diff(A, 2, axis=0)).max(axis=1)      # rad / tick^2 of the arm command
            return float(d2.max()), int((d2 > 0.03).sum())
        step, ramp_ = jerk(1), jerk(15)                         # fade_ticks 1 = the old one-tick removal
        self.assertLess(ramp_[0], step[0])
        self.assertLess(ramp_[1], step[1])

    def test_rl_with_each_fault(self):
        for faults, i, key, val in (({"miss": [0]}, 0, "grasp", "grasp_failed"),
                                    ({"drop": [1]}, 1, "transit", "lost_in_transit"),
                                    ({"stuck": [0]}, 0, "release", "recovered")):
            r = run_episode(faults, rl=True)
            self.assertEqual(r["clips"], 0, faults)
            self.assertEqual(r["pol"].grasp_log[i][key], val, faults)
            # RL closes deeper and is sampled per tick: allow the detection + opening time (< 1 s) over a bin
            self.assertLessEqual(r["empty_over_bin"], 30, faults)
            self.assertEqual(r["stuck_at_home"], 0, faults)


if __name__ == "__main__":
    unittest.main()


class TestGraspSelection(unittest.TestCase):
    """Issue 2 (wedged between index and middle): the grasp type follows from object width vs finger spacing."""

    @classmethod
    def setUpClass(cls):
        from aaltonomous.baselines.classical import grasp_select
        from aaltonomous.baselines.classical.config import hand_postures
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        from aaltonomous.scene.scene import Scene
        cls.G, cls.cfg, cls.sc = grasp_select, load_config(), Scene.load(None)
        cls.hand = grasp_select.hand_geometry(ArmKinematics(cls.cfg), hand_postures(cls.cfg),
                                              cls.cfg["grasp"]["select"]["pad_width"])

    def sel(self, shape, width=None):
        d = dict(self.sc.object_dims(shape))
        if width is not None:
            d["side"] = d["side"] * width / d["width"]; d["width"] = width
        return self.G.select(shape, d, self.hand, self.cfg["grasp"]["select"])

    def test_hand_geometry_comes_from_the_urdf(self):
        self.assertAlmostEqual(self.hand["spacing"], 0.057, places=3)     # index / middle lines
        self.assertAlmostEqual(self.hand["thumb"], 0.0, places=3)         # thumb closes midway between them

    def test_scene_objects_are_never_centred_between_the_two_fingers(self):
        for shape in ("cube", "cylinder", "triangle"):
            s = self.sel(shape)
            self.assertTrue(s["wedge_if_centred"], shape)                 # 45 mm on a 57 mm hand: would wedge
            self.assertEqual(s["type"], "pinch", shape)
            self.assertGreaterEqual(s["lateral"], 0.0085, shape)          # pinch finger on the flat face
            self.assertAlmostEqual(s["lateral"], 0.015, places=4)         # = the sim-validated grasp point

    def test_wide_object_gets_the_centred_three_finger_grasp(self):
        s = self.sel("cube", 0.090)
        self.assertEqual((s["type"], s["lateral"], s["wedge_if_centred"]), ("tripod", 0.0, False))

    def test_no_width_in_the_wedge_band_is_ever_centred(self):
        for w in np.arange(0.020, 0.100, 0.001):
            s = self.sel("cylinder", float(w))
            if s["wedge_if_centred"]:
                self.assertEqual(s["type"], "pinch", w)
                self.assertGreater(s["lateral"], 0.0, w)
            if s["type"] == "pinch":   # other finger clear of the object, pinch finger line on the object
                self.assertGreaterEqual(0.0285 + s["lateral"] - 0.009, w / 2 + 0.003 - 1e-9, w)
                self.assertLessEqual(0.0285 - s["lateral"], w / 2 + 1e-9, w)

    def test_planner_uses_the_selection_and_mid_height(self):
        r = run_episode()
        p = next(p for p in r["pol"].plans if p.ok)
        self.assertEqual(p.grasp["type"], "pinch")
        zc, H = r["env"].scene.counter_z, r["env"].scene.object_height(p.det.shape)
        self.assertGreaterEqual(p.waypoints["grasp"][2], zc + 0.5 * H - 1e-6)       # mid-height or fingertip-limited
        self.assertLess(p.waypoints["grasp"][2], zc + H - 0.008)                    # never above the object
