"""Validation sweep: baseline (t13 controller) vs a speed variant on randomized validation scenarios (live gc sim).
Each cell = one full episode (run_task.sh, inference only). The scoreboard is rewritten after EVERY episode:
runs/topdown_contact/speed/<tag>/scoreboard.{json,md}.
Run: python -m aaltonomous.experiments.topdown_contact.speed_sweep --tag v1 --variant runs/topdown_contact/variants/fast_approach_v1.json
     [--seeds 1000,1001,1002,1003] [--tasks sort_fixed,sort_language] [--reuse-baseline-1000]"""
import argparse, json, os, re, subprocess, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LIVE = REPO / "runs/topdown_contact/live"


def episode(name, task, seed, overrides):
    d = LIVE / name
    if not d.exists():
        env = {**os.environ, "AALTO_CFG_OVERRIDES": json.dumps(overrides)}
        subprocess.run(["bash", "aaltonomous/experiments/topdown_contact/run_task.sh", name, task, str(seed)], cwd=REPO, env=env,
                       stdout=open(REPO / f"runs/topdown_contact/logs/{name}.out", "w"), stderr=subprocess.STDOUT)
    return measure(d, task)


def measure(d, task):
    ev = sorted(d.glob("*/eval.json"))
    if not ev:
        return {"status": "no_eval"}
    t = json.loads(ev[-1].read_text())["trials"][0]
    picks = [json.loads(l) for l in (d / "td_picks.jsonl").read_text().splitlines()] if (d / "td_picks.jsonl").exists() else []
    log = (d / "policy_server.log").read_text() if (d / "policy_server.log").exists() else ""
    attempts = len(picks)
    ok = sum(1 for p in picks if p.get("outcome") == "released")
    lost = log.count("ABORT lost_in_transit")
    slips, rots = [], []
    for p in picks:
        D = p.get("diag") or []
        v = next((x for x in D if x["label"] == "verified"), None)
        aft = [x for x in D if v and x["t"] >= v["t"]]
        if v and aft:
            slips.append(max(float(np.linalg.norm(np.subtract(x["rel_mm"], v["rel_mm"]))) for x in aft))
    table = sum(1 for p in picks for e in (p.get("contact_events") or []) if e["partner"] != "object")
    good = "correct"
    outc = [o["outcome"] for o in t["outcomes"]]
    return {"task": task, "seed": t["seed"], "n_objects": len(outc), "n_correct": outc.count(good), "all_correct": bool(t["all_correct"]),
            "score": t["score"], "max_score": t["max_score"], "sim_s": t.get("sim_s") or t.get("wall_s"),
            "attempts": attempts, "grasp_ok": ok, "lost_in_transit": lost, "release_failed": sum(1 for p in picks if p.get("release") == "failed"),
            "slip_mm_max": round(max(slips), 1) if slips else None, "slip_mm_mean": round(float(np.mean(slips)), 1) if slips else None,
            "table_contacts": table, "collisions": sum(1 for p in picks if p.get("trigger") in ("collision", "proprio_block")),
            "clips": (t.get("safety") or {}).get("clipped_steps"), "end_reason": t.get("end_reason"), "gate_ok": t.get("gate_ok"),
            "gate_why": t.get("gate_why"), "dir": str(d)}


def agg(rows):
    rows = [r for r in rows if r.get("status") != "no_eval"]
    if not rows:
        return {}
    s = lambda k: sum(r[k] or 0 for r in rows)
    return {"episodes": len(rows), "placements": f"{s('n_correct')}/{s('n_objects')}", "tasks_all_correct": f"{sum(r['all_correct'] for r in rows)}/{len(rows)}",
            "grasp": f"{s('grasp_ok')}/{s('attempts')}", "lost_in_transit": s("lost_in_transit"), "release_failed": s("release_failed"),
            "table_contacts": s("table_contacts"), "collisions": s("collisions"), "clips": s("clips"),
            "gate_ok": f"{sum(bool(r['gate_ok']) for r in rows)}/{len(rows)}",
            "sim_s_mean": round(float(np.mean([r["sim_s"] for r in rows])), 1),
            "slip_mm_max": max([r["slip_mm_max"] for r in rows if r["slip_mm_max"] is not None] or [None]) if any(r["slip_mm_max"] is not None for r in rows) else None}


def write(out, cells, meta):
    board = {"meta": meta, "updated": time.strftime("%F %T"), "cells": cells, "summary": {}}
    for cfg in ("baseline", "variant"):
        for task in meta["tasks"]:
            board["summary"][f"{cfg}|{task}"] = agg([c for c in cells if c["cfg"] == cfg and c["task"] == task])
    # paired speed-up on seeds both configs finished
    for task in meta["tasks"]:
        pairs = []
        for seed in meta["seeds"]:
            b = next((c for c in cells if c["cfg"] == "baseline" and c["task"] == task and c["seed"] == seed and c.get("sim_s")), None)
            v = next((c for c in cells if c["cfg"] == "variant" and c["task"] == task and c["seed"] == seed and c.get("sim_s")), None)
            if b and v:
                pairs.append((b["sim_s"], v["sim_s"]))
        if pairs:
            bs, vs = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
            board["summary"][f"speedup|{task}"] = {"paired_seeds": len(pairs), "baseline_sim_s": round(bs, 1), "variant_sim_s": round(vs, 1),
                                                   "time_reduction_pct": round(100 * (1 - vs / bs), 1)}
    (out / "scoreboard.json").write_text(json.dumps(board, indent=1, default=str))
    L = [f"# Speed sweep {meta['tag']} ({board['updated']})", "", f"variant: `{meta['variant']}` {json.dumps(meta['overrides'])}", "",
         "| cfg | task | seed | correct | sim s | grasp ok/attempts | lost | table contacts | collisions | clips | gate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if c.get("status") == "no_eval":
            L.append(f"| {c['cfg']} | {c['task']} | {c['seed']} | NO EVAL | | | | | | | |"); continue
        L.append(f"| {c['cfg']} | {c['task']} | {c['seed']} | {c['n_correct']}/{c['n_objects']} | {c['sim_s']} | {c['grasp_ok']}/{c['attempts']} | "
                 f"{c['lost_in_transit']} | {c['table_contacts']} | {c['collisions']} | {c['clips']} | {'ok' if c['gate_ok'] else 'FAIL ' + ','.join(c['gate_why'] or [])} |")
    L += ["", "## Summary", "", "```", json.dumps(board["summary"], indent=1), "```"]
    (out / "scoreboard.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-len(board['summary']) * 3 - 8:]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", default="1000,1001,1002,1003"); ap.add_argument("--tasks", default="sort_fixed,sort_language")
    a = ap.parse_args()
    over = json.load(open(REPO / a.variant))
    seeds = [int(s) for s in a.seeds.split(",")]; tasks = a.tasks.split(",")
    out = REPO / f"runs/topdown_contact/speed/{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    meta = {"tag": a.tag, "variant": a.variant, "overrides": over, "seeds": seeds, "tasks": tasks, "sim": "gc (live WebRTC)"}
    existing = {("baseline", "sort_fixed", 1000): "task1_sort_fixed_seed1000", ("baseline", "sort_language", 1000): "task2_sort_language_seed1000"}
    cells = []
    for seed in seeds:                       # interleaved: a regression shows up early
        for task in tasks:
            for cfg, ov in (("variant", over), ("baseline", {})):
                name = existing.get((cfg, task, seed)) or f"{a.tag}_{cfg}_{task}_seed{seed}"
                r = episode(name, task, seed, ov) if not (LIVE / name).exists() else measure(LIVE / name, task)
                cells.append({"cfg": cfg, "task": task, "seed": seed, **r})
                write(out, cells, meta)


if __name__ == "__main__":
    main()
