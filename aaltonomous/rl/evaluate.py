"""Evaluate the scripted grasp (residual = 0) or a trained policy in the RL training env, or benchmark throughput.

    # scripted replay over the whole library, no placement noise, nominal physics (round-0 validation)
    python -m aaltonomous.rl.evaluate --policy scripted --cases library --physics nominal --out ...json
    # a checkpoint on held-out physics
    python -m aaltonomous.rl.evaluate --policy runs/rl/<id>/model_3000.pt --run aaltonomous/rl/runs/<id>.yaml \
        --physics fric0.3 --cases library --noise spec
    python -m aaltonomous.rl.evaluate --bench 256,1024,4096

Physics presets (fixed, not randomised): nominal (object 1.2/1.0, asset finger material, default gains, 50 g),
fric0.3 / fric0.9 (object AND fingers at that static friction), mass1.5, fingers_random (the randomised finger model,
fixed seed), fingers_default. Run on GPU 1: CUDA_VISIBLE_DEVICES=1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PHYSICS = {
    "nominal": dict(friction_mode="narrow", friction_fixed=1.2, mass_scale=(1.0, 1.0), finger_model="default"),
    "train_narrow": dict(friction_mode="narrow", friction_fixed=None, mass_scale=(0.7, 1.5), finger_model="default"),
    "fric0.3": dict(friction_fixed=0.3, mass_scale=(1.0, 1.0)),
    "fric0.9": dict(friction_fixed=0.9, mass_scale=(1.0, 1.0)),
    "mass1.5": dict(friction_fixed=1.2, mass_scale=(1.5, 1.5)),
    "fingers_random": dict(friction_fixed=1.2, mass_scale=(1.0, 1.0), finger_model="random"),
    "fingers_default": dict(friction_fixed=1.2, mass_scale=(1.0, 1.0), finger_model="default"),
    "dds_a0": dict(friction_fixed=1.2, fingers_match=False, mass_scale=(1.0, 1.0), finger_model="default"),  # = eval sim
    "run": {},          # the run config's own training distribution
}
NOISE = {
    "none": dict(est_pos_noise=0.0, est_yaw_noise_deg=0.0, est_wrong_shape_p=0.0, joint_noise=0.0),
    "low": dict(est_pos_noise=0.002, est_yaw_noise_deg=4.0, est_wrong_shape_p=0.0, joint_noise=0.001),
    "jitter1": dict(est_pos_noise=0.001, est_yaw_noise_deg=1.0, est_wrong_shape_p=0.0, joint_noise=0.0),
    "spec": dict(est_pos_noise=0.005, est_yaw_noise_deg=10.0, est_wrong_shape_p=0.05, joint_noise=0.003),
    "high": dict(est_pos_noise=0.010, est_yaw_noise_deg=20.0, est_wrong_shape_p=0.10, joint_noise=0.01),
    "run": {},          # the run config's own noise
}


def launch(num_envs, overrides, device="cuda:0"):
    from aaltonomous.rl import _paths  # noqa: F401
    from aaltonomous.rl.grasp_env import GraspEnv, GraspEnvCfg
    from isaaclab.app import launch_simulation
    cfg = GraspEnvCfg(num_envs=num_envs, **{k: v for k, v in overrides.items() if k != "num_envs"})
    cfg.sim.device = device
    ctx = launch_simulation(cfg, {"headless": True, "device": device, "require_kit": True})
    ctx.__enter__()
    env = GraspEnv(cfg)
    return env, ctx


def load_policy(path, env, run_cfg):
    import torch
    from aaltonomous.rl.train import make_runner
    runner = make_runner(env, run_cfg, log_dir=None)
    runner.load(path, load_optimizer=False)
    pol = runner.get_inference_policy(device=env.device)
    return lambda obs: pol(obs)


def run_cases(env, cases, policy=None, seed=0):
    """cases: dict of tensors entry, shape, dxy (n,2), dyaw (n,). Returns per-case result dicts."""
    import torch
    N, n = env.num_envs, len(cases["entry"])
    out = []
    torch.manual_seed(seed)
    for b0 in range(0, n, N):
        sl = slice(b0, min(n, b0 + N))
        k = sl.stop - sl.start
        ids = torch.arange(N, device=env.device)
        ent = torch.cat([cases["entry"][sl], cases["entry"][sl][:1].repeat(N - k)])
        shp = torch.cat([cases["shape"][sl], cases["shape"][sl][:1].repeat(N - k)])
        dxy = torch.cat([cases["dxy"][sl], torch.zeros(N - k, 2)])
        dyaw = torch.cat([cases["dyaw"][sl], torch.zeros(N - k)])
        env.reset()
        env._reset_episodes(ids, ent, shp, (dxy, dyaw))
        env.ep_stats, env.collect_stats = [], True
        obs = env._get_observations()
        first = {}
        for _ in range(env.T_lib + 3):
            with torch.inference_mode():
                if policy is None:
                    act = torch.zeros(N, env.cfg.action_space, device=env.device)
                else:
                    act = policy(obs)
            obs, *_ = env.step(act)
            for rec in env.ep_stats:
                for j in range(len(rec["env_id"])):
                    e = int(rec["env_id"][j])
                    if e < k and e not in first:
                        first[e] = {kk: (v[j].item() if hasattr(v[j], "item") else v[j]) for kk, v in rec.items()}
            env.ep_stats = []
            if len(first) >= k:
                break
        print(f"[eval] batch {b0 // N + 1}/{(n + N - 1) // N}: {sum(1 for v in first.values() if v.get('success'))}"
              f"/{k} success", flush=True)
        for e in range(k):
            r = first.get(e, {"success": False, "missing": True})
            r["case"] = b0 + e
            out.append(r)
    env.collect_stats = False
    return out


def library_cases(env, repeats=1, noise=None, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    n = env.n_lib
    ent = torch.arange(n).repeat(repeats)
    shp = env.lib["shape"].cpu()[ent]
    m = len(ent)
    nz = noise or {}
    pn, yn, wp = nz.get("est_pos_noise", 0.0), math.radians(nz.get("est_yaw_noise_deg", 0.0)), nz.get("est_wrong_shape_p", 0.0)
    dxy = (torch.rand(m, 2, generator=g) * 2 - 1) * pn
    dyaw = (torch.rand(m, generator=g) * 2 - 1) * yn
    if wp > 0:
        wrong = torch.rand(m, generator=g) < wp
        shp = torch.where(wrong, (shp + torch.randint(1, 3, (m,), generator=g)) % 3, shp)
    return {"entry": ent, "shape": shp, "dxy": dxy, "dyaw": dyaw}


def summarize(env, res, cases):
    import numpy as np
    from aaltonomous.rl.scripted import SHAPES
    L = {k: v.cpu().numpy() for k, v in env.lib.items() if k in ("x", "y", "yaw_pelvis", "shape")}
    rows = []
    for r in res:
        c = r["case"]
        e = int(cases["entry"][c])
        rows.append(dict(entry=e, shape=SHAPES[int(cases["shape"][c])], est_shape=SHAPES[int(L["shape"][e])],
                         x=float(L["x"][e]), y=float(L["y"][e]),
                         yaw_mat_deg=float(np.degrees(-np.pi / 2 - L["yaw_pelvis"][e])),
                         success=bool(r.get("success")), lifted_any=bool(r.get("lifted_any")),
                         held_hold=bool(r.get("held_hold")), violated=bool(r.get("violated")),
                         counter_ticks=float(r.get("counter_ticks", 0)), min_obj_dz=float(r.get("min_obj_dz", 0)),
                         final_rise=float(r.get("final_rise", 0)), missing=bool(r.get("missing", False))))
    by = {}
    for s in SHAPES:
        rr = [r for r in rows if r["shape"] == s]
        by[s] = {"n": len(rr), "success": sum(r["success"] for r in rr), "lifted_any": sum(r["lifted_any"] for r in rr),
                 "violated": sum(r["violated"] for r in rr), "missing": sum(r["missing"] for r in rr)}
    return rows, by


def bench(sizes, steps=300, device="cuda:0"):
    import torch
    out = {}
    for n in sizes:
        env, ctx = launch(n, {}, device)
        env.reset()
        for _ in range(20):
            env.step(torch.zeros(n, env.cfg.action_space, device=env.device))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.monotonic()
        for _ in range(steps):
            env.step(2 * torch.rand(n, env.cfg.action_space, device=env.device) - 1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.monotonic() - t0
        out[n] = {"env_steps_per_s": n * steps / dt, "ticks_per_s": steps / dt,
                  "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(f"[bench] {n} envs: {out[n]['env_steps_per_s']:.0f} env-steps/s ({out[n]['ticks_per_s']:.1f} ticks/s)",
              flush=True)
        env.close()
        break   # one sim context per process; the caller loops over sizes in separate processes
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="scripted")
    ap.add_argument("--run", default=None, help="run config yaml (env overrides + agent)")
    ap.add_argument("--physics", default="nominal")
    ap.add_argument("--noise", default="none")
    ap.add_argument("--cases", default="library")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=1560)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate every k-th case so that at most this many run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--lib", default=None, help="library npz (default lib_left.npz)")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu (the DDS eval sim runs PhysX on CPU)")
    a = ap.parse_args(argv)
    if a.bench:
        res = bench([int(v) for v in a.bench.split(",")])
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        return
    import yaml
    run = yaml.safe_load(open(a.run)) if a.run else {}
    ov = dict(run.get("env", {}))
    ov.update(PHYSICS[a.physics])
    ov.update(NOISE[a.noise])
    ov["seed"] = a.seed
    if a.lib:
        ov["lib_path"] = a.lib
    env, ctx = launch(a.num_envs, ov, a.device)
    pol = None if a.policy == "scripted" else load_policy(a.policy, env, run)
    cases = library_cases(env, a.repeats, NOISE.get(a.noise, {}) if a.noise != "run" else
                          {k: ov[k] for k in ("est_pos_noise", "est_yaw_noise_deg", "est_wrong_shape_p") if k in ov},
                          seed=a.seed)
    if a.limit and a.limit < len(cases["entry"]):
        step = len(cases["entry"]) // a.limit
        cases = {k: v[::step][: a.limit] for k, v in cases.items()}
    t0 = time.monotonic()
    res = run_cases(env, cases, pol, seed=a.seed)
    rows, by = summarize(env, res, cases)
    print(f"[eval] {a.policy} physics={a.physics} noise={a.noise}: {time.monotonic() - t0:.0f} s")
    for s, v in by.items():
        print(f"  {s:9s} success {v['success']}/{v['n']}  lifted_any {v['lifted_any']}  violated {v['violated']}"
              f"  missing {v['missing']}")
    meta = dict(policy=a.policy, run=a.run, physics=a.physics, noise=a.noise, repeats=a.repeats, seed=a.seed,
                env_overrides=ov, by_shape=by, robot_mat_nominal=env.robot_mat_nominal[:4].tolist(),
                finger_nominal=env.finger_nominal)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        import gzip
        op = gzip.open if a.out.endswith(".gz") else open
        with op(a.out, "wt") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=0, default=str)
    env.close()


if __name__ == "__main__":
    main()
