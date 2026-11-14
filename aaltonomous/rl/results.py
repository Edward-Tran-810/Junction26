"""Append evaluations to aaltonomous/rl/results.jsonl with the fields of
baselines/classical/variants/results.jsonl (one row per case), plus `env`:
  env = "rl_train_env": the RL training env (success = lifted >= 10 cm, held 1 s, still held after the shake;
        outcome "correct" means that, NOT a bin placement), or "dds": sim/grasp_test.py in the DDS eval sim.

    python -m aaltonomous.rl.results --eval evals/x.json.gz --lib data/lib_grasptest.npz --tag r0 \
        --variant none --assumption dds_a0 --policy scripted_oracle
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results.jsonl"


def _commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=HERE) != 0
        return c + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "?"


def failure(r):
    if r.get("violated"):
        return "keepout_violation"
    if not r.get("lifted_any"):
        return "never_lifted"
    if not r.get("held_hold"):
        return "dropped_in_hold"
    if not r.get("success"):
        return "dropped_in_shake"
    return "ok"


def env_rows(eval_path, lib_path, tag, variant, assumption, policy, noise=None):
    from .scripted import SHAPES
    d = json.load(gzip.open(eval_path, "rt") if str(eval_path).endswith(".gz") else open(eval_path))
    L = np.load(lib_path)
    has_cells = "cell_ymat" in L.files
    rep = {}
    out, ts, commit = [], time.strftime("%Y-%m-%dT%H:%M:%S"), _commit()
    for r in d["rows"]:
        e = r["entry"]
        yaw = float(np.degrees(L["yaw_mat"][e]))
        if has_cells:
            xm, ym = float(L["cell_ymat"][e]), 0.155
            arm, color = str(L["cell_arm"][e]), str(L["cell_color"][e])
        else:
            xm, ym = r["y"] + 0.45, r["x"] - 0.14      # pelvis -> mat
            arm, color = "left", ""
        shape = r["shape"]
        cell = f"{shape}/{yaw:.0f}/{xm:.3f}/{ym:.3f}"
        k = rep[cell] = rep.get(cell, -1) + 1
        out.append(dict(variant=variant, assumption=assumption, policy=policy, instance="rl_train_env", tag=tag,
                        cell=cell, shape=shape, yaw=round(yaw, 1), x=round(xm, 3), y=round(ym, 3), color=color,
                        repeat=k, outcome="correct" if r["success"] else "failed", failure=failure(r), clips=0,
                        command_hz=30.0, end="episode", arm=arm, est_shape=r.get("est_shape"),
                        noise=noise or d["meta"].get("noise"), env="rl_train_env",
                        physics=d["meta"].get("physics"), commit=commit, ts=ts))
    return out


def append(rows):
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="none")
    ap.add_argument("--assumption", required=True)
    ap.add_argument("--policy", default="scripted_oracle")
    a = ap.parse_args()
    append(env_rows(a.eval, a.lib, a.tag, a.variant, a.assumption, a.policy))
