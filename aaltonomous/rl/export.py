"""Export a checkpoint for the classical pipeline hook: <out>/policy.pt (TorchScript actor incl. obs normaliser, CPU)
+ <out>/skill_meta.json (env settings the skill must replicate), then time CPU inference.

    CUDA_VISIBLE_DEVICES=1 python -m aaltonomous.rl.export --ckpt runs/rl/<run>/model_399.pt --out runs/rl/<run>/export
    AALTO_RL_GRASP=runs/rl/<run>/export/policy.pt python -m aaltonomous.sim.grasp_test ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ck = Path(a.ckpt)
    out = Path(a.out or ck.parent / f"export_{ck.stem}")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(ck.parent / "meta.json"))
    run = meta["run"]
    from aaltonomous.rl.evaluate import launch
    from aaltonomous.rl.train import make_runner
    env, ctx = launch(8, dict(run.get("env") or {}))
    runner = make_runner(env, run, None)
    runner.load(str(ck), map_location=str(env.device))
    runner.export_policy_to_jit(path=str(out), filename="policy.pt")
    smeta = {"checkpoint": str(ck), "run_id": run.get("id"), "seed": meta.get("seed"),
             "env": {k: getattr(env.cfg, k) for k in ("action_space_name", "residual_arm", "residual_hand",
                                                      "use_estimate", "action_ema")},
             "obs_dim": env.cfg.observation_space, "act_dim": env.cfg.action_space}
    (out / "skill_meta.json").write_text(json.dumps(smeta, indent=1))
    env.close()
    import torch
    torch.set_num_threads(1)
    net = torch.jit.load(str(out / "policy.pt"), map_location="cpu").eval()
    x = torch.randn(1, smeta["obs_dim"])
    with torch.inference_mode():
        for _ in range(50):
            net(x)
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter()
            net(x)
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    smeta["cpu_infer_ms"] = {"median": float(np.median(ts)), "p99": float(np.percentile(ts, 99)),
                             "max": float(ts.max()), "threads": 1}
    (out / "skill_meta.json").write_text(json.dumps(smeta, indent=1))
    print(f"[export] {out}/policy.pt  CPU 1 thread: median {smeta['cpu_infer_ms']['median']:.3f} ms, "
          f"p99 {smeta['cpu_infer_ms']['p99']:.3f} ms")


if __name__ == "__main__":
    main()
