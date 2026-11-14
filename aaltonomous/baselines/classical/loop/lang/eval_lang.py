"""Score parse_instruction on frozen instruction batches.

    python -m aaltonomous.baselines.classical.loop.lang.eval_lang round_01.jsonl      # one fresh batch
    python -m aaltonomous.baselines.classical.loop.lang.eval_lang --regression        # regression.jsonl + templates
    python -m aaltonomous.baselines.classical.loop.lang.eval_lang --templates         # 8 templates x 36 + fixed

Batch line: {"id", "text", "gt": {"red/cube": "left" | null, ... all 9 types}, "category", "adversarial"}.
gt may also use colour / shape / default keys; it is expanded with scene.resolve_bin semantics.
Per instruction, over the 9 object types:
  exact   = every type's bin right (incl. "leave" = unmapped),
  wrong   = any type sent to a bin other than the true one (incl. a bin for a type that should stay),
  partial = not wrong, but some types the instruction maps are left unmapped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from aaltonomous.baselines.classical.instruction import COLORS, SHAPES, expand, parse_debug, parse_instruction

HERE = Path(__file__).resolve().parent


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def judge(pred_map, gt):
    p = expand(pred_map)
    g = expand(gt)
    wrong = [(k, p[k], g[k]) for k in g if p[k] is not None and p[k] != g[k]]
    missing = [(k, p[k], g[k]) for k in g if p[k] is None and g[k] is not None]
    if wrong:
        return "wrong", wrong + missing
    if missing:
        return "partial", missing
    return "exact", []


def template_rows():
    from aaltonomous.interface import tasks
    rows = [{"id": "fixed", "text": tasks.FIXED_INSTRUCTION, "gt": dict(tasks.FIXED_MAPPING), "category": "fixed"}]
    for ti, tpl in enumerate(tasks.TEMPLATES):
        for colors in itertools.permutations(tasks.COLORS):
            for bins in itertools.permutations(tasks.BINS):
                w = {f"c{i}": c for i, c in enumerate(colors)}
                w.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
                w.update({f"b{i}": b for i, b in enumerate(bins)})
                rows.append({"id": f"tpl{ti}_{''.join(c[0] for c in colors)}_{''.join(b[0] for b in bins)}",
                             "text": tpl.format(**w), "gt": dict(zip(colors, bins)), "category": "template"})
    return rows


PARSER = parse_instruction      # A1: --parser rule | lm | hybrid swaps this
_CACHE = {}


def _parse(text):
    if text not in _CACHE:
        _CACHE[text] = PARSER(text)
    return _CACHE[text]


def score(rows, verbose=True, label=""):
    counts = {"exact": 0, "partial": 0, "wrong": 0}
    fails = []
    for r in rows:
        pred = _parse(r["text"])
        verdict, detail = judge(pred, r["gt"])
        counts[verdict] += 1
        if verdict != "exact":
            fails.append((r, verdict, detail, pred))
    n = len(rows)
    split = {}
    for r in rows:
        v = judge(_parse(r["text"]), r["gt"])[0]
        d = split.setdefault("adv" if r.get("adversarial") else "non-adv", {"exact": 0, "partial": 0, "wrong": 0})
        d[v] += 1
    if verbose:
        for r, verdict, detail, pred in fails:
            print(f"--- [{verdict}] {r.get('id')} ({r.get('category', '')}{', ADV' if r.get('adversarial') else ''})")
            print(f"    text: {r['text']}")
            p = parse_debug(r["text"]) if PARSER is parse_instruction else None
            if p is not None:
                print(f"    norm: {p.norm}")
            print(f"    diff (type, got, true): {[(f'{c}/{s}', a, b) for (c, s), a, b in detail]}")
            if p is not None and p.notes:
                print(f"    notes: {p.notes}")
    print(f"{label}n={n} exact={counts['exact']} ({100 * counts['exact'] / max(n, 1):.1f}%) "
          f"partial={counts['partial']} wrong={counts['wrong']}"
          + ("  | " + "; ".join(f"{k}: n={sum(d.values())} exact={d['exact']} partial={d['partial']} wrong={d['wrong']}"
                                for k, d in sorted(split.items())) if "adv" in split else ""))
    return counts, fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="*")
    ap.add_argument("--regression", action="store_true", help="every absorbed batch + templates")
    ap.add_argument("--templates", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--parser", choices=("rule", "lm", "hybrid"), default="rule")
    a = ap.parse_args(argv)
    global PARSER
    if a.parser != "rule":
        from . import lm_parser
        PARSER = lm_parser.parse_lm if a.parser == "lm" else lm_parser.parse_hybrid
    ok = True
    for b in a.batches:
        path = Path(b) if Path(b).exists() else HERE / b
        c, _ = score(load(path), not a.quiet, f"[{path.name}] ")
    if a.templates or a.regression:
        c, _ = score(template_rows(), not a.quiet, "[templates 8x36+fixed] ")
        ok &= c["exact"] == sum(c.values())
    if a.regression:
        reg = HERE / "regression.jsonl"
        if reg.exists():
            c, _ = score(load(reg), not a.quiet, "[regression] ")
            ok &= c["exact"] == sum(c.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
