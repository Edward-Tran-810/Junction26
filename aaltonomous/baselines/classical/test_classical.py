"""Offline tests of the classical baseline (no robot). Run:
    ~/miniforge3/envs/aalto/bin/python -m unittest aaltonomous/baselines/classical/test_classical.py -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config, mirror_left_to_right
from .kinematics import ArmKinematics
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .planner import Planner
from .policy import ClassicalPolicy

SC, CAL, CFG = Scene.load(), Calibration.load(), load_config()


def scene_objects(rng, n=3, shapes=("cube", "cylinder")):
    L = SC.cfg["layout"]
    objs, used = [], []
    colors = list(rng.permutation(["red", "yellow", "blue"]))[:n]
    while len(objs) < n:
        x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
        if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
            continue
        used.append((x, y))
        h = SC.object_height()
        objs.append(dict(color=colors[len(objs)], shape=shapes[len(objs) % len(shapes)], yaw=rng.uniform(-0.8, 0.8),
                         size=SC.size, height=h, pos=SC.mat_to_pelvis([x, y, h / 2])))
    return objs


class TestPerception(unittest.TestCase):
    def setUp(self):
        self.r = Renderer(SC, CAL)
        self.p = Perception(CFG, SC, CAL)

    def _check(self, rgb, depth, objs, tol):
        dets = self.p.detect(rgb, depth)
        self.assertEqual(sorted(d.color for d in dets), sorted(o["color"] for o in objs))
        for o in objs:
            d = next(d for d in dets if d.color == o["color"])
            top = o["pos"] + [0, 0, o["height"] / 2]
            err = np.linalg.norm(d.top - top)
            self.assertLess(err, tol, f"{o['color']} {o['shape']} err {err * 1000:.1f} mm ({d.method})")
        return dets

    def test_top_centre_within_1cm_with_noise_and_holes(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            objs = scene_objects(rng)
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.5, hole_frac=0.15)
            dets = self._check(rgb, depth, objs, 0.01)
            self.assertTrue(all(d.method == "depth" for d in dets))

    def test_no_depth_falls_back_to_plane(self):
        rng = np.random.default_rng(1)
        objs = scene_objects(rng)
        rgb, _ = self.r.render(objs)
        dets = self._check(rgb, np.zeros(rgb.shape[:2], np.uint16), objs, 0.025)
        self.assertTrue(all(d.method == "plane" for d in dets))

    def test_bins_are_not_objects(self):
        rgb, depth = self.r.render([])
        self.assertEqual(self.p.detect(rgb, depth), [])


class TestPostures(unittest.TestCase):
    def test_mirror_and_limits(self):
        post = hand_postures(CFG)  # raises if outside joint_spec limits
        for v in post.values():
            np.testing.assert_allclose(v[7:], mirror_left_to_right(v[:7]))
        kin = ArmKinematics(CFG)
        for name, v in post.items():  # mirrored postures put the right fingertips at the mirrored place
            np.testing.assert_allclose(kin.tips_in_palm("right", v), kin.tips_in_palm("left", v) * [1, -1, 1], atol=1e-6)


class TestIK(unittest.TestCase):
    def test_reaches_points_check_layout_calls_reachable(self):
        """Points over the mat that check_layout's random palm cloud reaches (palm 8 cm above an object centre)
        must be solved by our IK (palm frame, any orientation) within 1 cm. (cloud tolerance 1 cm, not the 3 cm
        check_layout prints with, so 'reachable' means a sampled configuration really got there.)"""
        from aaltonomous.scene.check_layout import palm_cloud, reach_map
        cfg = load_config(overrides={"grasp.point_in_palm": [0.0, 0.0, 0.0]})
        kin = ArmKinematics(cfg)
        home = home_arm(cfg)
        rng = np.random.default_rng(0)
        W, D = SC.cfg["counter"]["mat_size"]
        mat = np.c_[rng.uniform(0, W, 300), rng.uniform(0, D, 300), np.full(300, SC.object_height() / 2)]
        P = SC.mat_to_pelvis(mat)
        ok = reach_map(SC, palm_cloud(40000), P, 0.08, 0.01)
        pts = P[ok][:40] + [0, 0, 0.08]
        self.assertGreater(len(pts), 10)
        hit = 0
        for p in pts:
            side = "left" if p[1] > 0 else "right"
            res = [kin.solve(s, p, home, top_down=False) for s in (side, "right" if side == "left" else "left")]
            hit += min(r[2] for r in res) < 0.01
        self.assertGreaterEqual(hit / len(pts), 0.95, f"{hit}/{len(pts)}")

    def test_top_down_grasp_over_cube_region(self):
        kin, home = ArmKinematics(CFG), home_arm(CFG)
        L = SC.cfg["layout"]
        n = ok = 0
        for x in np.linspace(*L["region_x"], 5):
            for y in np.linspace(*L["region_y"], 3):
                p = SC.mat_to_pelvis([x, y, 0.5 * SC.object_height()])
                n += 1
                ok += any(kin.solve(s, p, home)[1] for s in ("left", "right"))
        self.assertGreaterEqual(ok / n, 0.8, f"top-down grasp IK {ok}/{n} over the cube region")


class TestPlanner(unittest.TestCase):
    def test_plan_respects_speed_and_safety(self):
        from .perception import Detection
        post, home = hand_postures(CFG), home_arm(CFG)
        pl = Planner(CFG, SC, post, home)
        L = SC.cfg["layout"]
        top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])
        p = pl.plan(Detection("yellow", top, 0, 0, "depth"), home, post["open"])
        self.assertTrue(p.ok, p.reason)
        dq = np.abs(np.diff(np.vstack([home, p.arm]), axis=0)).max()
        self.assertLessEqual(dq, CFG["motion"]["arm_max_vel"] / 30 + 1e-9)
        dh = np.abs(np.diff(np.vstack([post["open"], p.hand]), axis=0)).max()
        self.assertLessEqual(dh, CFG["motion"]["hand_max_vel"] / 30 + 1e-9)
        self.assertEqual(p.safety_clips, 0)
        np.testing.assert_allclose(p.arm[-1], home, atol=1e-9)  # ends at home


    def test_cross_body_object_is_relocated(self):
        """Relocation fallback: the right arm puts a red (left bin) object down on the counter where the left arm has a
        full plan from home. With the published box row the left arm reaches the whole object band directly, so the
        fallback is called explicitly here."""
        from .perception import Detection
        cfg = load_config(profile="none")
        post, home = hand_postures(cfg), home_arm(cfg)
        pl = Planner(cfg, SC, post, home)
        top = SC.mat_to_pelvis([0.30, 0.18, SC.object_height()])
        det = Detection("red", top, 0, 0, "depth")
        direct = pl.plan(det, home, post["open"])
        self.assertTrue(direct.ok, direct.reason)
        self.assertEqual((direct.side, direct.target_bin), ("left", "left"))
        p = pl.plan_relocation(det, "left", home, post["open"])
        self.assertTrue(p.ok, p.reason)
        self.assertEqual((p.side, p.target_bin), ("right", "counter"))
        self.assertEqual(p.safety_clips, 0)
        spot = p.waypoints["drop"]
        self.assertLess(abs(spot[2] - p.waypoints["grasp"][2]), 0.01)  # put down at grasp height, not dropped
        fin = pl.plan(Detection("red", np.r_[spot[:2], top[2]], 0, 0, "depth"), home, post["open"], relocate=False)
        self.assertTrue(fin.ok, fin.reason)
        self.assertEqual((fin.side, fin.target_bin), ("left", "left"))


class TestEpisode(unittest.TestCase):
    def test_full_mock_episodes(self):
        total_correct = 0
        for seed in (0, 1, 2):
            env = make_mock_env(seed=seed)
            pol = ClassicalPolicy(log=None, safety=env.filter.cfg)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            while True:
                obs, info = env.step(pol.act(obs))
                if info["episode_over"]:
                    break
            self.assertIn(info["end_reason"], ("done", "time_limit"))
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
            out = env.backend.outcomes(pol.mapping)
            self.assertNotIn("wrong", out)
            self.assertNotIn("off", out)
            total_correct += out.count("correct")
        self.assertGreaterEqual(total_correct, 3)

    def test_confirm_gate_holds_until_confirmed(self):
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, require_confirm=True)
        obs = env.reset()
        pol.reset()
        obs, info = env.step(pol.act(obs))
        self.assertTrue(pol.waiting)          # first: confirm the move to home
        self.assertIn("home", pol.pending_text)
        pol.confirm()
        for _ in range(400):
            obs, info = env.step(pol.act(obs))
            if pol.waiting:
                break
        self.assertTrue(pol.waiting)          # then: confirm the first object
        self.assertIsNotNone(pol.pending)
        q = obs["arm_q"].copy()
        k = int(pol.cfg["motion"]["hold_ticks"])
        for _ in range(60 // k):  # 2 s of waiting: arms hold (k-tick hold chunks), no watchdog stop
            a = pol.act(obs)
            self.assertEqual(a["arm_q_target"].shape, (k, 14))
            self.assertEqual(np.ptp(a["arm_q_target"], axis=0).max(), 0.0)
            obs, info = env.step(a)
        self.assertEqual(info["safety_mode"], "ok")
        self.assertLess(np.abs(obs["arm_q"] - q).max(), 1e-3)
        pol.confirm()
        obs, info = env.step(pol.act(obs))
        self.assertFalse(pol.waiting)
        self.assertEqual(pol.state, "executing")


if __name__ == "__main__":
    unittest.main()


class TestInstruction(unittest.TestCase):
    """Rung-2 instruction parser (instruction.py) and how the policy uses its mapping."""

    def test_every_training_template_and_the_fixed_sentence(self):
        from .loop.lang.eval_lang import judge, template_rows
        from .instruction import parse_instruction
        bad = [r["text"] for r in template_rows() if judge(parse_instruction(r["text"]), r["gt"])[0] != "exact"]
        self.assertEqual(bad, [])

    def test_frozen_regression_set(self):
        from .loop.lang import eval_lang
        reg = eval_lang.HERE / "regression.jsonl"
        if not reg.exists():
            self.skipTest("no regression.jsonl yet")
        from .instruction import parse_instruction
        bad = [r["text"] for r in eval_lang.load(reg) if eval_lang.judge(parse_instruction(r["text"]), r["gt"])[0] != "exact"]
        self.assertEqual(bad, [])

    def test_shape_dependent_colour_is_left_when_shape_unknown(self):
        from .instruction import parse_instruction, target_bin
        m = parse_instruction("Red cubes go left, all other red things go right, yellow and blue in the middle.")
        self.assertEqual(target_bin(m, "red", "cube"), "left")
        self.assertEqual(target_bin(m, "red", "triangle"), "right")
        self.assertIsNone(target_bin(m, "red"))            # vision without shape: never guess
        self.assertEqual(target_bin(m, "blue"), "center")

    def test_unknown_colour_never_falls_through_to_default(self):
        from .instruction import target_bin
        m = {"red": "left", "default": "right"}
        self.assertIsNone(target_bin(m, None, "cube"))      # could be red (left) or not (right): never guess
        self.assertIsNone(target_bin(m, None))
        self.assertEqual(target_bin({"cube": "left", "default": "right"}, None, "cube"), "left")   # shape decides
        self.assertEqual(target_bin({"default": "center"}, None, None), "center")                  # all agree

    def test_unclear_parts_stay_unmapped(self):
        from .instruction import parse_instruction
        m = parse_instruction("Red in the xyz bin, yellow left, blue right.")
        self.assertEqual(m, {"yellow": "left", "blue": "right"})
        self.assertEqual(parse_instruction("Leave the blue ones on the counter, the rest go right."),
                         {"red": "right", "yellow": "right"})

    def test_policy_mapping_sources(self):
        from aaltonomous.interface.tasks import FIXED_MAPPING
        pol = ClassicalPolicy.__new__(ClassicalPolicy)   # reset() only touches the planner's mapping
        pol.planner = type("P", (), {})()
        pol.time_limit = 120.0
        pol.reset({"instruction": "Blue goes left, the rest in the right bin.", "task": "sort_language"})
        self.assertEqual(pol.mapping, {"red": "right", "yellow": "right", "blue": "left"})
        pol.reset({"instruction": "Blue goes left, the rest in the right bin.", "task": "sort_fixed"})
        self.assertEqual(pol.mapping, FIXED_MAPPING)
        pol.reset({"instruction": "gibberish", "task": "sort_language"})
        self.assertEqual(pol.mapping, {})                  # not the fixed mapping: nothing is guessed
        pol.reset({"mapping": {}})
        self.assertEqual(pol.mapping, {})

    def test_swaps_and_double_negatives(self):
        from .instruction import parse_instruction
        self.assertEqual(parse_instruction("Do the usual sorting, but switch the yellow and the blue."),
                         {"red": "left", "yellow": "right", "blue": "center"})
        self.assertEqual(parse_instruction("Red left, yellow middle, blue right, then reverse everything."),
                         {"red": "right", "yellow": "center", "blue": "left"})       # reverse = mirror left/right
        self.assertEqual(parse_instruction("Red left, yellow middle, blue right, then rotate them."), {})  # unreadable
        self.assertEqual(parse_instruction("Never leave a blue object on the table; blue belongs in the left bin."),
                         {"blue": "left"})
        self.assertEqual(parse_instruction("No yellow in the right bin: yellow goes in the centre."),
                         {"yellow": "center"})


class TestReLook(unittest.TestCase):
    """A5 self-check with known truth: the post-place re-look reports where the picked object went."""

    def _episode(self, attach):
        from .loop.closed_loop import instructions, run_episode
        item = next(i for i in instructions() if i["kind"] == "template")
        return run_episode(501, item, attach_radius=attach, time_limit=60.0)

    def test_missed_grasp_is_seen_on_the_counter(self):
        r = self._episode(0.0)                     # the mock hand never holds anything
        self.assertTrue(r["place_checks"])
        self.assertTrue(all(c["outcome"] == "still_on_counter" for c in r["place_checks"]), r["place_checks"])
        self.assertTrue(all(o == "counter" for o in r["outcomes"]), r["outcomes"])

    def test_successful_place_leaves_the_counter(self):
        r = self._episode(0.04)
        self.assertTrue(r["place_checks"])
        self.assertTrue(all(c["outcome"] == "left_counter" for c in r["place_checks"]), r["place_checks"])
        self.assertEqual(r["score"], r["max_score"])
