"""PPO (rsl_rl) on Aalto-Grasp-G129-Dex3-RL. One run = one config file aaltonomous/rl/runs/<id>.yaml + a seed.

    CUDA_VISIBLE_DEVICES=1 python -m aaltonomous.rl.train --run aaltonomous/rl/runs/<id>.yaml --seed 1
Logs + checkpoints: runs/rl/<id>_s<seed>/ (repo-root runs/ is gitignored). Asymmetric: the actor sees the
`policy` group (what a submission has), the critic the `critic` group (true state).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
AGENT_DEFAULT = dict(
    num_steps_per_env=32, max_iterations=3000, save_interval=100, empirical_normalization=False,
    obs_groups={"actor": ["policy"], "critic": ["critic"]},
    actor=dict(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True, init_noise_std=0.3),
    critic=dict(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True),
    algorithm=dict(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.002,
                   num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)


def agent_cfg(run_cfg, seed, device):
    from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,
                                    handle_deprecated_rsl_rl_cfg)
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version
    a = json.loads(json.dumps(AGENT_DEFAULT))
    for k, v in (run_cfg.get("agent") or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k].update(v)
        else:
            a[k] = v
    std = a["actor"].pop("init_noise_std")
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed, device=device, num_steps_per_env=a["num_steps_per_env"], max_iterations=a["max_iterations"],
        save_interval=a["save_interval"], empirical_normalization=a["empirical_normalization"],
        obs_groups=a["obs_groups"], experiment_name="aalto_grasp", init_at_random_ep_len=False,
        actor=RslRlMLPModelCfg(**a["actor"], distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=std)),
        critic=RslRlMLPModelCfg(**a["critic"]),
        algorithm=RslRlPpoAlgorithmCfg(**a["algorithm"]),
    )
    return handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version()), a


def make_runner(env, run_cfg, log_dir, seed=0):
    from isaaclab.utils import to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    cfg, _ = agent_cfg(run_cfg, seed, str(env.device))
    w = RslRlVecEnvWrapper(env, clip_actions=None)
    return OnPolicyRunner(w, to_dict(cfg), log_dir=log_dir, device=str(env.device))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--log-root", default=str(REPO / "runs/rl"))
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    a = ap.parse_args(argv)
    run = yaml.safe_load(open(a.run))
    rid = Path(a.run).stem
    if a.max_iterations:
        run.setdefault("agent", {})["max_iterations"] = a.max_iterations
    n = a.num_envs or int(run.get("num_envs", 4096))
    log_dir = Path(a.log_root) / f"{rid}_s{a.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    from aaltonomous.rl.evaluate import launch
    ov = dict(run.get("env") or {})
    ov["seed"] = a.seed
    env, ctx = launch(n, ov)
    runner = make_runner(env, run, str(log_dir), a.seed)
    if a.resume:
        runner.load(a.resume)
    json.dump({"run": run, "seed": a.seed, "num_envs": n, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(log_dir / "meta.json", "w"), indent=1)
    t0 = time.monotonic()
    runner.learn(num_learning_iterations=int(run.get("agent", {}).get("max_iterations", AGENT_DEFAULT["max_iterations"])),
                 init_at_random_ep_len=False)
    (log_dir / "done.json").write_text(json.dumps({"wall_s": time.monotonic() - t0}))
    env.close()


if __name__ == "__main__":
    main()
