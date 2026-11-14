"""Evaluate checkpoints as they appear (round 1: model_200 and model_399) in the training env, deterministic actions.

own  : the run's own training distribution (physics run, noise run), 2 x library = ~3120 cases, seed 7. The
       scripted baseline on the SAME cases + physics draws is the kill-rule reference.
held : (last checkpoint only) held-out physics with the run's own noise: fric0.3, fric0.9, mass1.5, the other finger
       model, dds_a0 (eval-sim finger materials, mu 5). Scripted baselines per (condition, noise) are cached.
Summaries: evals/summary_<tag>.jsonl (one line per evaluation, raw counts per shape).

    python -m aaltonomous.rl.autoeval --tag r1 --ckpts 200,399 [--once]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "aaltonomous/rl/evals"
RUNS = REPO / "runs/rl"


def held_conditions(run_env):
    other = "fingers_default" if run_env.get("finger_model") == "random" else "fingers_random"
    return ["fric0.3", "fric0.9", "mass1.5", other, "dds_a0"]


def run_eval(policy, run_yaml, physics, out, seed=7, repeats=2, noise="run"):
    if out.exists():
        return json.load(gzip.open(out, "rt"))["meta"]
    env = dict(os.environ, PYTHONPATH=str(REPO), OMNI_KIT_ACCEPT_EULA="YES", CUDA_VISIBLE_DEVICES="1")
    cmd = [sys.executable, "-m", "aaltonomous.rl.evaluate", "--policy", policy, "--run", str(run_yaml),
           "--physics", physics, "--noise", noise, "--repeats", str(repeats), "--num-envs", "3120",
           "--seed", str(seed), "--out", str(out)]
    log = out.with_suffix(".log")
    r = subprocess.run(cmd, env=env, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode != 0 or not out.exists():
        print(f"[autoeval] FAILED {cmd} (see {log})", flush=True)
        return None
    log.unlink(missing_ok=True)
    return json.load(gzip.open(out, "rt"))["meta"]


def noise_key(env):
    return "n{}_{}_{}_{}_e{}".format(env.get("est_pos_noise"), env.get("est_yaw_noise_deg"), env.get("est_wrong_shape_p"),
                                     env.get("joint_noise"), int(bool(env.get("use_estimate", True))))


def scripted(run_yaml, physics, run_env, tag):
    """Scripted baseline. Own distribution: per run config (physics draws depend on it). Held-out: per noise."""
    if physics == "run":
        out = EV / f"{tag}_scripted_{Path(run_yaml).stem}_own.json.gz"
    else:
        fm = run_env.get("finger_model", "default") if physics in ("fric0.3", "fric0.9", "mass1.5") else ""
        out = EV / f"{tag}_scripted_{physics}_{fm}_{noise_key(run_env)}.json.gz"
    return run_eval("scripted", run_yaml, physics, out)


def total(meta):
    return sum(v["success"] for v in meta["by_shape"].values())


def evaluate_ckpt(run_dir, ck, tag, held):
    meta = json.load(open(run_dir / "meta.json"))
    rid = run_dir.name.rsplit("_s", 1)[0]
    run_yaml = REPO / f"aaltonomous/rl/runs/{rid}.yaml"
    run_env = meta["run"].get("env", {})
    conds = ["run"] + (held_conditions(run_env) if held else [])
    lines = []
    for c in conds:
        out = EV / f"{tag}_{run_dir.name}_m{ck}_{c}.json.gz"
        m = run_eval(str(run_dir / f"model_{ck}.pt"), run_yaml, c, out)
        b = scripted(run_yaml, c, run_env, tag)
        if m is None or b is None:
            continue
        line = {"tag": tag, "run": run_dir.name, "ckpt": ck, "physics": c, "noise": "run",
                "policy": {s: [v["success"], v["n"]] for s, v in m["by_shape"].items()},
                "scripted": {s: [v["success"], v["n"]] for s, v in b["by_shape"].items()},
                "total": [total(m), total(b)], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(EV / f"summary_{tag}.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        p, s = line["policy"], line["scripted"]
        print(f"[autoeval] {run_dir.name} m{ck} {c:15s} " + "  ".join(
            f"{k} {p[k][0]}/{p[k][1]} (scr {s[k][0]})" for k in p) + f"  total {line['total'][0]} vs {line['total'][1]}",
              flush=True)
        lines.append(line)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--ckpts", default="200,399")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    cks = [int(c) for c in a.ckpts.split(",")]
    done = set()
    sp = EV / f"summary_{a.tag}.jsonl"
    if sp.exists():
        for l in open(sp):
            d = json.loads(l)
            done.add((d["run"], d["ckpt"], d["physics"]))
    while True:
        did = False
        for rd in sorted(Path(p) for p in glob.glob(str(RUNS / "r*_s*")) if Path(p).is_dir()):
            for ck in cks:
                if not (rd / f"model_{ck}.pt").exists() or (rd.name, ck, "run") in done:
                    continue
                for l in evaluate_ckpt(rd, ck, a.tag, held=(ck == cks[-1])):
                    done.add((l["run"], l["ckpt"], l["physics"]))
                did = True
        if a.once:
            break
        if not did:
            time.sleep(120)


if __name__ == "__main__":
    main()
