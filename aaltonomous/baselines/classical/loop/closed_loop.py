"""PERCEPTION_MILESTONES A5: parser + perception + policy together in the mock world, over a FIXED seed list.

    python -m aaltonomous.baselines.classical.loop.closed_loop                   # 40 episodes (default list)
    python -m aaltonomous.baselines.classical.loop.closed_loop --n 10 --set '{"perception.colour.mode": "wb_proto"}'

Episode k: seed SEEDS[k] (layout), instruction INSTR[k] = the k-th item of a fixed list mixing the frozen language
batches round_09 + round_10 (true 9-type mapping in `gt`; non-adversarial and adversarial) and the training
templates. The policy only sees the TEXT (instruction.py parses it); the episode is scored on the TRUE mapping
with scene.score (+2 correct bin, -1 wrong bin, 0 left on the counter). Also reported: the re-look outcome of every
executed pick (policy.place_checks) against the mock world's own truth after the episode.

MOCK WORLD: kinematic grasping, simple rendering (mock_world.py). Exercises the logic chain, not physics or
appearance: not a stop-rule result.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = list(range(500, 540))


def instructions():
    rows = []
    for name in ("round_09.jsonl", "round_10.jsonl"):
        rows += [json.loads(line) for line in (HERE / "lang" / name).read_text().splitlines() if line.strip()]
    # interleave: 3 batch items (stride 7 through the 120) then 1 training template, fixed order
    from aaltonomous.interface import tasks
    import numpy as np
    rng = np.random.default_rng(12345)
    out, i = [], 0
    while len(out) < 60:
        for _ in range(3):
            r = rows[(i * 7) % len(rows)]
            out.append({"id": r["id"], "text": r["text"], "gt": r["gt"], "kind": "adv" if r.get("adversarial") else "batch"})
            i += 1
        text, mapping = tasks.make_instruction("sort_language", rng)
        out.append({"id": f"tmpl_{len(out)}", "text": text, "gt": mapping, "kind": "template"})
    return out


def run_episode(seed, item, overrides=None, attach_radius=0.04, time_limit=120.0):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.instruction import mapping_from_json
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    from aaltonomous.scene.scene import resolve_bin
    cfg = load_config(None, overrides or {})
    env = make_mock_env(seed=seed, cfg=cfg, task="sort_language", attach_radius=attach_radius)
    env.time_limit = time_limit
    env.fixed_instruction = item["text"]
    obs = env.reset()
    truth = mapping_from_json(item["gt"])
    truth = {k: v for k, v in truth.items() if v is not None}
    env.mapping = truth
    pol = ClassicalPolicy(config=cfg, safety=env.filter.cfg, require_confirm=False, threaded=False,
                          time_limit=time_limit, log=lambda *a: None)
    pol.reset({"instruction": obs["instruction"], "task": "sort_language"})
    info = {}
    t0 = time.time()
    while True:
        obs, info = env.step(pol.act(obs))
        if info["episode_over"]:
            break
    outc = env.backend.outcomes(truth)
    score = env.scene.score("sort_language", outc)
    return {"seed": seed, "id": item["id"], "kind": item["kind"], "score": score, "outcomes": outc,
            "n_obj": len(outc), "end": info.get("end_reason"), "t_sim": round(info.get("t", 0.0), 1),
            "clipped": info.get("safety_summary", {}).get("clipped_steps") if isinstance(info.get("safety_summary"), dict) else None,
            "place_checks": pol.place_checks, "wall_s": round(time.time() - t0, 1),
            "max_score": 2 * sum(1 for o in env.backend.objects if resolve_bin(truth, o["color"], o.get("shape")))}


def _job(args):
    return run_episode(*args)


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--set", default=None, help="config overrides (json)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ov = json.loads(a.set) if a.set else None
    items = instructions()
    jobs = [(SEEDS[k], items[k], ov) for k in range(a.n)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = list(pool.imap(_job, jobs))
    tot, mx = sum(r["score"] for r in res), sum(r["max_score"] for r in res)
    from collections import Counter
    oc = Counter(o for r in res for o in r["outcomes"])
    chk = Counter(c["outcome"] for r in res for c in r["place_checks"])
    print(f"mock world, {len(res)} episodes (seeds {SEEDS[0]}..{SEEDS[a.n - 1]}): score {tot} of max {mx} "
          f"({100 * tot / max(mx, 1):.1f} %), mean {tot / len(res):.2f} / episode")
    print(f"  object outcomes: {dict(oc)}")
    print(f"  re-look outcomes: {dict(chk)}")
    for kind in ("batch", "adv", "template"):
        rs = [r for r in res if r["kind"] == kind]
        if rs:
            print(f"  {kind:<9} n={len(rs):>2} score {sum(r['score'] for r in rs)} / {sum(r['max_score'] for r in rs)}")
    for r in res:
        if r["score"] < r["max_score"]:
            print(f"   seed {r['seed']} {r['id']:<10} {r['kind']:<8} score {r['score']}/{r['max_score']} "
                  f"{r['outcomes']} end={r['end']} t={r['t_sim']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
