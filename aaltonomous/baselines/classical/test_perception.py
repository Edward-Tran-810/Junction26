"""Offline tests of the classical perception (v2 pipeline, shape_fit, synthetic bench). No simulator, no robot.

    OMP_NUM_THREADS=4 python -m unittest aaltonomous/baselines/classical/test_perception.py -v

Fixtures are deterministic synthetic RGB-D frames (loop/vision/synth.py) with exact ground truth. They test that the
ALGORITHM does what it should under controlled disturbances; they say nothing about accuracy on Isaac or real
frames (see loop/STATE.md, vision track).
"""
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .instruction import target_bin
from .loop.vision.synth import SynthRenderer, sample_spec
from .perception import Detection, Perception
from .shape_fit import PERIOD, TopFaceFitter, outline, wrap_yaw

SC, CAL = Scene.load(), Calibration.load()
RENDER = SynthRenderer(SC, CAL, supersample=False)     # 1 sample / pixel: fast enough for unit tests
GATE = 0.03


def spec_with(objects, family="nominal", seed=0, **over):
    """A frame spec with hand-placed objects [(colour, shape, x_mat, y_mat, layout_yaw)] on top of a sampled spec
    (lighting, texture, sensor and camera error of `family`). `over` replaces top-level spec entries."""
    s = sample_spec(seed, family, "band", SC)
    s["objects"] = [{"color": c, "shape": sh, "x": x, "y": y, "yaw": yaw, "albedo": list(np.asarray(SC.colors[c]) / 255)}
                    for c, sh, x, y, yaw in objects]
    s["layout"] = [{k: o[k] for k in ("color", "shape", "x", "y", "yaw")} for o in s["objects"]]
    s.update(over)
    return s


def perception(**ov):
    """A fresh Perception WITHOUT the per-episode camera-correction cache (every fixture frame is its own scene)."""
    ov.setdefault("perception.refine_cache.enabled", False)
    return Perception(load_config(None, ov), SC, CAL)


def match(dets, gts):
    """Greedy nearest match (objects are far apart in these fixtures): list of (det, gt, xy error m)."""
    out, used = [], set()
    for g in gts:
        cands = [(np.hypot(*(d.top[:2] - np.asarray(g["top"][:2]))), i) for i, d in enumerate(dets) if i not in used]
        if cands:
            dist, i = min(cands)
            if dist < GATE:
                used.add(i)
                out.append((dets[i], g, dist))
    return out


def yaw_err_deg(det_yaw, gt_yaw, shape):
    p = np.degrees(PERIOD[shape])
    a = abs(np.degrees(det_yaw - gt_yaw)) % p
    return min(a, p - a)


THREE = [("red", "cube", 0.33, 0.16, 0.3), ("yellow", "cylinder", 0.45, 0.15, 0.0), ("blue", "triangle", 0.57, 0.16, -0.4)]


class TestSegmentationAndShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = perception()

    def test_colours_shapes_positions_nominal(self):
        rgb, depth, gt = RENDER.render(spec_with(THREE))
        dets = self.p.detect(rgb, depth)
        self.assertEqual(len(dets), 3, dets)
        m = match(dets, gt["objects"])
        self.assertEqual(len(m), 3)
        for d, g, err in m:
            self.assertEqual(d.color, g["color"])
            self.assertEqual(d.shape, g["shape"], d.fit_iou)
            self.assertLess(err, 0.003, f"{g['shape']} xy error {err * 1000:.1f} mm")
            self.assertAlmostEqual(d.top[2], g["top"][2], delta=0.004)
            self.assertEqual(d.method, "depth")

    def test_every_shape_at_several_yaws(self):
        for k, yaw in enumerate(np.radians([-40, -20, 0, 15, 35])):
            objs = [("red", "cube", 0.30, 0.15, yaw), ("yellow", "triangle", 0.45, 0.16, yaw),
                    ("blue", "cylinder", 0.60, 0.15, yaw)]
            rgb, depth, gt = RENDER.render(spec_with(objs, seed=k))
            for d, g, err in match(self.p.detect(rgb, depth), gt["objects"]):
                self.assertEqual(d.shape, g["shape"], f"yaw {np.degrees(yaw):.0f}: {d.fit_iou}")
                if g["shape"] in ("cube", "triangle"):
                    self.assertLess(yaw_err_deg(d.yaw, g["yaw"], g["shape"]), 3.0)
                else:
                    self.assertIsNone(d.yaw)

    def test_two_same_colour_objects_close_together(self):
        objs = [("blue", "cube", 0.40, 0.16, 0.2), ("blue", "triangle", 0.47, 0.16, 0.0), ("red", "cylinder", 0.60, 0.15, 0)]
        rgb, depth, gt = RENDER.render(spec_with(objs))
        dets = self.p.detect(rgb, depth)
        m = match(dets, gt["objects"])
        self.assertEqual(len(m), 3, dets)
        self.assertEqual(len(dets), 3)
        for d, g, _ in m:
            self.assertIn(d.shape, (g["shape"], None))      # never a wrong shape

    def test_empty_counter_and_bins_are_not_objects(self):
        rgb, depth, _ = RENDER.render(spec_with([]))
        self.assertEqual(self.p.detect(rgb, depth), [])

    def test_flat_stickers_in_object_colours_are_not_objects(self):
        st = [{"color": "red", "shape": "cube", "x": 0.30, "y": 0.12, "yaw": 0.3, "size": 0.05,
               "albedo": list(np.asarray(SC.colors["red"]) / 255)},
              {"color": "yellow", "shape": "triangle", "x": 0.62, "y": 0.18, "yaw": 0.0, "size": 0.05,
               "albedo": list(np.asarray(SC.colors["yellow"]) / 255)}]
        rgb, depth, gt = RENDER.render(spec_with([("blue", "cube", 0.45, 0.16, 0.0)], stickers=st))
        dets = self.p.detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["blue"], dets)

    def test_partly_hidden_object_never_gets_a_wrong_shape(self):
        objs = [("red", "cube", 0.40, 0.16, 0.2)]
        occ = [{"x": 0.40, "y": 0.13, "z0": 0.07, "size": [0.06, 0.05, 0.03], "rgb": [0.15] * 3}]
        rgb, depth, gt = RENDER.render(spec_with(objs, occluders=occ))
        dets = self.p.detect(rgb, depth)
        for d, g, _ in match(dets, gt["objects"]):
            self.assertIn(d.shape, ("cube", None), d.fit_iou)


class TestDepthHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = perception()

    def test_noise_and_holes(self):
        sensor = {"rgb_gain": 1.0, "rgb_noise": 4.0, "blur": 0.5, "depth_noise_mm": 2.0, "depth_noise_rel": 0.003,
                  "hole_frac": 0.15, "edge_invalid_p": 0.6, "stereo_band_px": 5, "depth_shift_px": [1, 0]}
        rgb, depth, gt = RENDER.render(spec_with(THREE, sensor=sensor))
        m = match(self.p.detect(rgb, depth), gt["objects"])
        self.assertEqual(len(m), 3)
        for d, g, err in m:
            self.assertIn(d.shape, (g["shape"], None))
            self.assertLess(err, 0.005)

    def test_no_depth_stream_falls_back_to_colour_and_plane(self):
        rgb, depth, gt = RENDER.render(spec_with(THREE))
        dets = self.p.detect(rgb, np.zeros_like(depth))
        m = match(dets, gt["objects"])
        self.assertEqual(len(m), 3, dets)
        for d, g, err in m:
            self.assertEqual(d.method, "plane")
            self.assertIn(d.shape, (g["shape"], None))
            self.assertLess(err, 0.01)               # nominal camera: render-and-compare on the known plane
        self.assertEqual(repr(self.p.detect(rgb, None)), repr(dets))   # depth None = no depth stream

    def test_depth_rays_match_calibration_deprojection(self):
        rays = self.p._pixel_rays((480, 640))
        T = CAL._T("head")
        rng = np.random.default_rng(0)
        u, v, d = rng.integers(0, 640, 50), rng.integers(0, 480, 50), rng.uniform(300, 900, 50)
        P_ref = CAL.pixel_to_pelvis("head", u, v, d)
        P = T[:3, 3] + (d / 1000.0)[:, None] * rays[:, v, u].T
        np.testing.assert_allclose(P, P_ref, atol=1e-5)
        uvz = CAL.project("head", P_ref)       # round trip; calibration.yaml stores R to 6 decimals (~1e-3 px)
        np.testing.assert_allclose(uvz[:, 0], u, atol=2e-3)
        np.testing.assert_allclose(uvz[:, 1], v, atol=2e-3)


class TestCalibration(unittest.TestCase):
    def test_camera_pose_error_is_corrected_by_plane_and_bins(self):
        cam = {"dt": [0.008, -0.009, 0.006], "drot_deg": [0.9, -0.8, 0.7]}
        rgb, depth, gt = RENDER.render(spec_with(THREE, camera=cam))
        errs_on = [e for _, _, e in match(perception().detect(rgb, depth), gt["objects"])]
        p_off = perception(**{"perception.plane_refine.enabled": False, "perception.bin_refine.enabled": False})
        errs_off = [e for _, _, e in match(p_off.detect(rgb, depth), gt["objects"])]
        self.assertEqual(len(errs_on), 3)
        self.assertLess(max(errs_on), 0.004)
        self.assertGreater(np.median(errs_off), 0.006)   # the refinement is what removes the error

    def test_refinement_reports_and_limits(self):
        p = perception()
        rgb, depth, gt = RENDER.render(spec_with(THREE))
        p.detect(rgb, depth)
        self.assertTrue(p.last_plane["applied"])
        self.assertTrue(p.last_plane["bins"]["applied"])
        self.assertLess(abs(p.last_plane["bins"]["yaw_deg"]), 0.3)
        # wrong wall height in the config: no rim points -> bin refinement not applied, detection still works
        sc2 = Scene.load()
        sc2.cfg["bins"]["wall_height"] = 0.10
        p2 = Perception(load_config(), sc2, CAL)
        dets = p2.detect(rgb, depth)
        self.assertFalse(p2.last_plane["bins"]["applied"])
        self.assertEqual(len(dets), 3)

    def test_missing_extrinsics_raise_a_clear_error(self):
        cal = Calibration.load()
        cal.cams["head"].T_pelvis_optical = None
        p = Perception(load_config(), SC, cal)
        rgb, depth, _ = RENDER.render(spec_with(THREE))
        with self.assertRaisesRegex(ValueError, "extrinsics"):
            p.detect(rgb, depth)


class TestRobustness(unittest.TestCase):
    def test_unseen_lighting_and_textures(self):
        """Held-out-style frames (procedural mats incl. wood, tinted / dim / harsh light, stickers): every visible
        object found, no false positives, no wrong colour or shape."""
        p = perception()
        for seed in (11, 12, 13, 14):
            rgb, depth, gt = RENDER.render(sample_spec(seed + 500, "unseen", "band", SC))
            dets = p.detect(rgb, depth)
            vis = [g for g in gt["objects"] if g["visible_px"] >= 0.5 * g["unoccluded_px"]]
            m = match(dets, vis)
            self.assertEqual(len(m), len(vis), f"seed {seed}: {dets}")
            for d, g, _ in m:
                self.assertEqual(d.color, g["color"])
                self.assertIn(d.shape, (g["shape"], None))

    def test_calibration_cache_is_per_episode(self):
        """With refine_cache on, the camera correction of the first frame is reused until reset_calibration()."""
        p = Perception(load_config(), SC, CAL)
        cam_a = {"dt": [0.008, -0.009, 0.006], "drot_deg": [0.9, -0.8, 0.7]}
        cam_b = {"dt": [-0.008, 0.009, -0.006], "drot_deg": [-0.9, 0.8, -0.7]}
        fa = RENDER.render(spec_with(THREE, camera=cam_a))
        fb = RENDER.render(spec_with(THREE, camera=cam_b))
        p.detect(*fa[:2])
        p.detect(*fb[:2])
        self.assertTrue(p.last_plane.get("cached"))                  # frame b used frame a's correction
        p.reset_calibration()
        errs = [e for _, _, e in match(p.detect(*fb[:2]), fb[2]["objects"])]
        self.assertFalse(p.last_plane.get("cached", False))
        self.assertLess(max(errs), 0.004)

    def test_repeatable_and_fast(self):
        import cv2
        cv2.setNumThreads(1)
        p = perception()
        rgb, depth, _ = RENDER.render(spec_with(THREE))
        a, b = p.detect(rgb, depth), p.detect(rgb, depth)
        self.assertEqual([(d.color, d.shape) for d in a], [(d.color, d.shape) for d in b])
        np.testing.assert_array_equal(np.array([d.top for d in a]), np.array([d.top for d in b]))
        t = []
        for _ in range(5):
            t0 = time.perf_counter()
            p.detect(rgb, depth)
            t.append(time.perf_counter() - t0)
        self.assertLess(np.median(t), 0.040)        # loose bound for shared machines; eval_vision reports the number


class TestInterfaces(unittest.TestCase):
    def test_detection_is_backward_compatible(self):
        d = Detection("red", np.array([0.3, 0.0, 0.055]), 100, 80, "depth")
        self.assertIsNone(d.shape)
        self.assertIsNone(d.yaw)
        self.assertIn("red", repr(d))

    def test_legacy_pipeline_still_runs(self):
        rgb, depth, gt = RENDER.render(spec_with(THREE))
        dets = perception(**{"perception.pipeline": "legacy"}).detect(rgb, depth)
        self.assertEqual(sorted(d.color for d in dets), ["blue", "red", "yellow"])
        self.assertTrue(all(d.shape is None for d in dets))

    def test_shape_feeds_the_bin_choice(self):
        mapping = {("red", "cube"): "left", ("red", "cylinder"): "right", ("red", "triangle"): "right"}
        self.assertEqual(target_bin(mapping, "red", "cube"), "left")
        self.assertIsNone(target_bin(mapping, "red", None))          # shape unknown: not sorted, never guessed
        self.assertEqual(target_bin({"red": "left"}, "red", None), "left")

    def test_yaw_conventions(self):
        self.assertAlmostEqual(wrap_yaw(np.radians(50), "cube"), np.radians(-40))
        self.assertAlmostEqual(wrap_yaw(np.radians(70), "triangle"), np.radians(-50))
        self.assertIsNone(wrap_yaw(0.3, "cylinder"))
        # canonical outline: triangle apex along +y at yaw 0, centred on the centroid
        tri = outline(SC, "triangle", 0.0)
        np.testing.assert_allclose(tri.mean(0), 0, atol=1e-12)
        self.assertAlmostEqual(tri[:, 1].max(), 2 * SC.object_dims("triangle")["apothem"])

    def test_top_face_fitter_on_exact_points(self):
        f = TopFaceFitter(SC)
        rng = np.random.default_rng(0)
        for shape in ("cube", "cylinder", "triangle"):
            yaw = 0.3
            poly = outline(SC, shape, yaw)
            pts = rng.uniform(-0.035, 0.035, (6000, 2))
            e = np.roll(poly, -1, 0) - poly
            inside = np.all([e[j, 0] * (pts[:, 1] - poly[j, 1]) - e[j, 1] * (pts[:, 0] - poly[j, 0]) >= 0
                             for j in range(len(poly))], axis=0)
            fit = f.fit(pts[inside] + [0.4, 0.1])
            self.assertEqual(fit.shape, shape)
            np.testing.assert_allclose(fit.xy, [0.4, 0.1], atol=0.0015)
            if shape != "cylinder":
                self.assertLess(yaw_err_deg(fit.yaw, yaw, shape), 2.0)


class TestRecordingTools(unittest.TestCase):
    def test_ground_truth_from_scene_state_matches_the_renderer(self):
        """capture_sim.gt_from_state (Isaac scene-service poses -> GT) agrees with synth.py's GT for the same
        layout: same top centre, same yaw (mod symmetry), projected area close to the rendered pixel count."""
        from aaltonomous.sim.aalto_task.layout import mat_yaw_to_pelvis, yaw_quat_xyzw
        from .loop.vision.capture_sim import gt_from_state
        spec = spec_with(THREE)
        _, _, gt = RENDER.render(spec)
        state = []
        for ob in spec["objects"]:
            h = SC.object_height(ob["shape"])
            state.append({"name": "x", "color": ob["color"], "shape": ob["shape"], "active": True, "speed": 0.0,
                          "pos": list(SC.mat_to_pelvis([ob["x"], ob["y"], h / 2])),
                          "quat_xyzw": list(yaw_quat_xyzw(mat_yaw_to_pelvis(ob["yaw"])))})
        rec = gt_from_state(state, SC, CAL)
        for r, g in zip(rec, gt["objects"]):
            np.testing.assert_allclose(r["top"], g["top"], atol=1e-6)
            if g["shape"] != "cylinder":
                self.assertLess(yaw_err_deg(r["yaw"], g["yaw"], g["shape"]), 1e-3)
            self.assertFalse(r["tipped"])
            self.assertAlmostEqual(r["visible_px"] / g["unoccluded_px"], 1.0, delta=0.05)

    def test_recorded_splits_are_disjoint(self):
        import json
        import tempfile
        from pathlib import Path
        from .loop.vision import dataset
        tmp_root = Path(dataset.HERE / "cache")            # stay inside the repo (gitignored), not /tmp
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as d:
            Path(d, "index.jsonl").write_text("".join(json.dumps({"id": f"isaac-band-{i}", "seed": i}) + "\n"
                                                      for i in range(420)))
            old, dataset.RECORDED = dataset.RECORDED, Path(d)
            try:
                parts = [dataset.split_ids("isaac_dev")] + [dataset.split_ids(f"isaac_fresh_{k:02d}")
                                                            for k in range(1, 7)] + [dataset.split_ids("isaac_holdout")]
            finally:
                dataset.RECORDED = old
        sizes = [len(x) for x in parts]
        self.assertEqual(sizes, [120] + [40] * 6 + [60])
        allids = [i for x in parts for i in x]
        self.assertEqual(len(set(allids)), 420)


class TestEndToEndMock(unittest.TestCase):
    def test_shape_instruction_sorted_in_the_mock_world(self):
        """instruction.py (shape rules) + v2 shapes + planner in the rendered mock world (true triangle prisms)."""
        from .mock_world import make_mock_env
        from .policy import ClassicalPolicy
        text = "Put the cubes in the left bin and the cylinders in the right bin. Triangles go in the middle bin."
        truth = {"cube": "left", "cylinder": "right", "triangle": "center"}
        env = make_mock_env(seed=2, task="sort_language")
        obs = env.reset()
        env.instruction, env.mapping = text, truth
        obs["instruction"] = text
        pol = ClassicalPolicy(config=load_config(), safety=env.filter.cfg, time_limit=120, log=lambda *a: None)
        pol.reset({"instruction": text, "task": "sort_language"})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        outc = env.backend.outcomes(truth)
        self.assertEqual(sorted({o["shape"] for o in env.backend.objects}), ["cube", "cylinder", "triangle"])
        self.assertEqual(outc, ["correct"] * len(outc))
        self.assertEqual(info["clipped_steps"], 0)


if __name__ == "__main__":
    unittest.main()


class TestHueHybrid(unittest.TestCase):
    """Detector + hue hybrid: joint per-frame colour assignment, self-check with known truth."""

    def test_order_decides_where_nearest_prototype_collides(self):
        from .loop.vision.learned import assign_colours
        sim = {"red": 0.0, "yellow": 25.0, "blue": 113.0}
        # real foam blocks (orange 12, yellow 41, purple 110) against the sim prototypes
        out = assign_colours([12.0, 13.5, 41.0, 110.0], sim)
        self.assertEqual([c for c, _ in out], ["red", "red", "yellow", "blue"])
        # orange 16 is nearer yellow (9) than red (16): alone it would collide with the yellow cluster
        out = assign_colours([16.0, 40.0, 109.0], sim, max_dist=30)
        self.assertEqual([c for c, _ in out], ["red", "yellow", "blue"])

    def test_wraparound_and_abstain(self):
        from .loop.vision.learned import assign_colours
        p = {"red": 0.0, "yellow": 25.0, "blue": 113.0}
        out = assign_colours([178.0, 2.0, None, 70.0], p)
        self.assertEqual(out[0][0], "red")
        self.assertEqual(out[1][0], "red")
        self.assertIsNone(out[2][0])
        self.assertIsNone(out[3][0])          # green-ish hue: 45 from yellow, 43 from blue -> abstain
