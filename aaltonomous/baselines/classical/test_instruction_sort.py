"""Sort-by-instruction tests: parsing (dev vs held-out kept apart), contradictions / corrections, and end to end in the
mock world (the PARSED mapping decides the bin; scoring uses the instruction's INTENDED mapping, not the parser's).

    python -m unittest aaltonomous/baselines/classical/test_instruction_sort.py

heldout_v1.jsonl was written before the parser was run on it and must not be used for tuning (loop/lang/make_heldout.py).
Parsing accuracy and physical sorting are reported separately: this file measures parsing and the mock-world routing.
"""
import itertools
import unittest
from pathlib import Path

from aaltonomous.baselines.classical.instruction import COLORS, expand, mapping_for_task, parse_instruction
from aaltonomous.baselines.classical.loop.lang.eval_lang import judge, load
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy

LANG = Path(__file__).parent / "loop" / "lang"


def colour_map(text):
    return {k[0]: v for k, v in expand(parse_instruction(text)).items() if v and k[1] == "cube"}


class TestHeldOut(unittest.TestCase):
    def test_heldout_paraphrases_and_distractors_exact(self):
        rows = [r for r in load(LANG / "heldout_v1.jsonl") if not r["adversarial"]]
        bad = [(r["id"], r["text"]) for r in rows if judge(parse_instruction(r["text"]), r["gt"])[0] != "exact"]
        self.assertEqual(bad, [])
        self.assertEqual(len(rows), 44)

    def test_heldout_adversarial_never_sends_an_unclear_colour_to_a_bin(self):
        for r in (r for r in load(LANG / "heldout_v1.jsonl") if r["adversarial"]):
            verdict = judge(parse_instruction(r["text"]), r["gt"])[0]
            if r["category"] == "contradiction_red":       # "...left bin. Actually no, ... right bin": a correction
                continue
            self.assertNotEqual(verdict, "wrong", (r["id"], r["text"]))

    def test_all_six_permutations_every_style(self):
        rows = [r for r in load(LANG / "heldout_v1.jsonl") if r["category"].startswith("perm")]
        self.assertEqual(len({tuple(sorted(r["gt"].items())) for r in rows}), 6)


class TestContradictionAndCorrection(unittest.TestCase):
    def test_same_colour_two_bins_is_not_guessed(self):
        for t in ("Put blue in the middle box and also in the right box. Red left, yellow center.",
                  "Red to the left bin, yellow to the center bin, blue to the middle bin and also the right bin."):
            m = colour_map(t)
            self.assertNotIn("blue", m, t)                       # unmapped = left on the counter, never a wrong bin
            self.assertEqual(m.get("red"), "left" if "Red to" in t or "Red left" in t else None, t)
            self.assertEqual(m.get("yellow"), "center", t)

    def test_clear_corrections_take_the_later_statement(self):
        for t in ("Put red on the left — actually, put it on the right. Yellow in the center, blue in the left.",
                  "Put the red objects in the left bin. Actually no, put the red objects in the right bin. "
                  "Yellow center, blue left."):
            self.assertEqual(colour_map(t), {"red": "right", "yellow": "center", "blue": "left"}, t)

    def test_unrelated_also_does_not_create_a_conflict(self):
        self.assertEqual(colour_map("Red left, yellow center, blue right; also put the red ones in the left bin too."),
                         {"red": "left", "yellow": "center", "blue": "right"})


class TestRoutingEndToEnd(unittest.TestCase):
    """Mock world: object -> bin follows the instruction, for every permutation, against the INTENDED mapping."""

    def run_ep(self, text, intended, seed=0):
        env = make_mock_env(seed=seed, task="sort_language")
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": text, "task": "sort_language"})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        return pol, env, env.backend.outcomes(intended), info

    def test_every_permutation_routes_to_the_instructed_bins(self):
        names = {"left", "center", "right"}
        for perm in itertools.permutations(sorted(names)):
            m = dict(zip(("red", "yellow", "blue"), perm))
            text = f"Please put red things in the {m['red']} box, blue things in the {m['blue']} box and yellow things in the {m['yellow']} box."
            pol, env, out, info = self.run_ep(text, m)
            self.assertEqual({c: pol.mapping[c] for c in COLORS}, m)
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
            self.assertNotIn("wrong", out, m)
            self.assertGreaterEqual(out.count("correct"), 3, (m, out))

    def test_default_mapping_is_not_used_for_language_tasks(self):
        text = "Red into the right box, yellow into the left box, blue into the center box."
        pol, env, out, _ = self.run_ep(text, {"red": "right", "yellow": "left", "blue": "center"})
        from aaltonomous.interface.tasks import FIXED_MAPPING
        self.assertNotEqual(pol.mapping, FIXED_MAPPING)
        self.assertNotIn("wrong", out)

    def test_contradicted_colour_is_left_on_the_counter_not_misplaced(self):
        text = "Put blue in the middle box and also in the right box. Red goes left, yellow goes center."
        pol, env, out, _ = self.run_ep(text, {"red": "left", "yellow": "center"})
        self.assertIsNone(pol.mapping.get("blue"))
        self.assertTrue(all(g["color"] != "blue" for g in pol.grasp_log))     # never even picked
        self.assertNotIn("wrong", out)

    def test_sort_fixed_still_ignores_the_text(self):
        self.assertEqual(mapping_for_task("sort_fixed", "blue left"), {"red": "left", "yellow": "center", "blue": "right"})


if __name__ == "__main__":
    unittest.main()
