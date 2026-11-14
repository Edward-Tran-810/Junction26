"""Rung-3 insertion: geometry + planning tests (offline; no simulator).

The REAL lid holes are outside the arms' reach (see GRASP_FIXES.md / test_real_holes_reachability_is_reported), so
the planner tests use the same geometry with the holes moved toward the robot ("synthetic") to exercise the planning code
(turn, slow insertion, lid clearance). That is NOT evidence the real insertion works.
"""
import unittest

import numpy as np

from aaltonomous.baselines.classical.insertion import SHAPE_TO_HOLE, SYMMETRY, InsertGeometry
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.scene.scene import Scene


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = Scene.load(None)
        cls.g = InsertGeometry(cls.sc)

    def test_shape_to_hole_mapping_is_by_shape_not_colour(self):
        self.assertEqual(SHAPE_TO_HOLE, {"cube": "square", "cylinder": "circle", "triangle": "triangle"})
        for s in ("cube", "cylinder", "triangle"):
            self.assertEqual(self.g.hole_for(s).shape, s)
        self.assertIsNone(self.g.hole_for(None))             # unknown shape: never a guess
        self.assertIsNone(self.g.hole_for("sphere"))

    def test_holes_come_from_the_scene_and_are_clearance_larger_than_the_objects(self):
        for s, h in self.g.holes.items():
            self.assertAlmostEqual(h.apothem - self.sc.object_dims(s)["apothem"], self.sc.cfg["lid"]["hole_clearance"] / 2, places=9)
            self.assertTrue(self.g.fits(s, h.centre, h.yaw))
        self.assertAlmostEqual(self.g.lid_top_z - self.g.lid_under_z, self.sc.cfg["lid"]["thickness"], places=9)

    def test_tolerances_measured_from_the_polygons(self):
        for s in ("cylinder", "cube", "triangle"):
            self.assertAlmostEqual(self.g.tolerance(s)["xy_m"], 0.005, delta=0.0006)       # = clearance / 2
        self.assertAlmostEqual(np.degrees(self.g.tolerance("cube")["yaw_rad"]), 14.7, delta=0.6)
        self.assertAlmostEqual(np.degrees(self.g.tolerance("triangle")["yaw_rad"]), 11.8, delta=0.6)

    def test_wrong_orientation_or_wrong_hole_does_not_fit(self):
        h = self.g.holes
        self.assertFalse(self.g.fits("cube", h["cube"].centre, h["cube"].yaw + np.pi / 4))      # square turned 45 deg
        self.assertFalse(self.g.fits("triangle", h["triangle"].centre, h["triangle"].yaw + np.pi / 3))   # apex flipped
        self.assertFalse(self.g.fits("cube", h["triangle"].centre, 0.0, hole_shape="triangle"))
        # a cylinder DOES pass through the square hole (scores wrong_hole): matching by shape is mandatory
        self.assertTrue(self.g.fits("cylinder", h["cube"].centre, 0.0, hole_shape="cube"))

    def test_align_delta_is_the_smallest_turn_within_the_symmetry(self):
        for s in ("cube", "triangle"):
            h, per = self.g.holes[s], SYMMETRY[s]
            for d in np.radians([-40, -20, 0, 7, 25, 44, 89]):
                delta = self.g.align_delta(s, h.yaw + d)
                self.assertLessEqual(abs(delta), per / 2 + 1e-9)
                self.assertTrue(self.g.fits(s, h.centre, h.yaw + d + delta))           # turning by delta makes it fit
        self.assertEqual(self.g.align_delta("cylinder", 0.7), 0.0)

    def test_pose_error_budget_is_reported_not_assumed(self):
        rng = np.random.default_rng(1)
        p = {}
        for s in ("cube", "triangle"):
            h = self.g.holes[s]
            ok = sum(self.g.fits(s, h.centre + rng.uniform(-5e-3, 5e-3, 2), h.yaw + np.radians(rng.uniform(-10, 10))) for _ in range(600))
            p[s] = ok / 600
        self.assertLess(p["cube"], 0.7)        # at the project's nominal noise spec a cube mostly does NOT fit
        self.assertLess(p["triangle"], 0.7)    # => insertion needs a pose refinement step (documented in the report)


class TestPlanner(unittest.TestCase):
    """Planning code on synthetic (reachable) holes."""

    @classmethod
    def setUpClass(cls):
        cls.pol = ClassicalPolicy(log=None, threaded=False)
        cls.pl, cls.sc = cls.pol.planner, cls.pol.scene
        cls.pl.insert = InsertGeometry(cls.sc)
        sh = np.array([-0.09, 0.0])
        for h in cls.pl.insert.holes.values():
            h.centre, h.poly = h.centre + sh, h.poly + sh

    def det(self, shape, yaw, xy=(0.34, 0.14)):
        return Detection("red", np.r_[xy, self.sc.counter_z + self.sc.object_height()], 1000, 1000, "oracle", yaw=yaw, shape=shape)

    def test_unknown_shape_is_rejected_for_a_re_look(self):
        p = self.pl.plan_insert(self.det(None, None), self.pol.home.copy(), self.pol.post["open"].copy())
        self.assertFalse(p.ok)
        self.assertIn("shape unknown", p.reason)

    def test_cube_plan_turns_to_the_hole_and_inserts_slowly(self):
        d = self.det("cube", np.radians(20))
        p = self.pl.plan_insert(d, self.pol.home.copy(), self.pol.post["open"].copy())
        self.assertTrue(p.ok, p.reason)
        names = [n for n, _, _ in p.phases]
        self.assertEqual(names[:3], ["to_pregrasp", "descend", "close"])
        self.assertLess(names.index("lift"), names.index("turn"))
        self.assertLess(names.index("carry"), names.index("insert"))
        self.assertLess(names.index("insert"), names.index("open"))
        self.assertAlmostEqual(abs(p.grasp["turn_deg"]), 20.0, delta=0.5)               # turned by what the hole needs
        self.assertLessEqual(p.grasp["turn_err_deg"], self.pol.cfg["insert"]["yaw_tol_deg"])
        self.assertEqual(p.safety_clips, 0)
        ph = {n: (s, e) for n, s, e in p.phases}
        hole = self.pl.insert.holes["cube"]
        np.testing.assert_allclose(p.waypoints["drop"][:2], hole.centre, atol=1e-9)    # hole pose, NO drop-spread offset
        self.assertEqual(p.target_bin, "lid")
        # the insertion descent is slower than the carry
        arm = p.arm
        v_ins = np.abs(np.diff(arm[ph["insert"][0]:ph["insert"][1]], axis=0)).max()
        v_car = np.abs(np.diff(arm[ph["carry"][0]:ph["carry"][1]], axis=0)).max()
        self.assertLess(v_ins, v_car)

    def test_target_is_independent_of_how_many_objects_were_dropped_elsewhere(self):
        self.pl.drop_count = {"left": 3, "center": 2}
        p = self.pl.plan_insert(self.det("cylinder", None), self.pol.home.copy(), self.pol.post["open"].copy())
        self.assertTrue(p.ok, p.reason)
        np.testing.assert_allclose(p.waypoints["drop"][:2], self.pl.insert.holes["cylinder"].centre, atol=1e-9)
        self.pl.drop_count = {}

    def test_object_bottom_ends_at_the_lid_underside_with_fingers_above_the_lid(self):
        p = self.pl.plan_insert(self.det("cylinder", None), self.pol.home.copy(), self.pol.post["open"].copy())
        self.assertTrue(p.ok, p.reason)
        g, H = self.pl.insert, self.sc.object_height("cylinder")
        self.assertAlmostEqual(p.waypoints["drop"][2] - H / 2, g.lid_under_z + self.pol.cfg["insert"]["end_gap"], places=6)

    def test_real_holes_reachability_is_reported(self):
        """The REAL hole centres: the left arm's grasp point cannot get there (documented limitation, measured)."""
        pol = ClassicalPolicy(log=None, threaded=False)
        g = InsertGeometry(pol.scene)
        errs = {}
        for s, h in g.holes.items():
            _, ok, err, _ = pol.kin.solve("left", np.r_[h.centre, 0.12], pol.home, top_down=True)
            errs[s] = (ok, err)
        self.assertFalse(all(ok for ok, _ in errs.values()), errs)       # if this starts failing: reach changed -> run Isaac


if __name__ == "__main__":
    unittest.main()
