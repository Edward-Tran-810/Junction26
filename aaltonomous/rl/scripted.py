"""Scripted grasp trajectories for the RL env: the oracle's planner (classical planner + ORACLE_OVERRIDES +
face-aligned grasp, imported, not edited) planned for the LEFT arm on one object, cut to
pregrasp -> descend -> close -> lift, then a hold and a scripted shake appended here.

Pure CPU (pinocchio). Planning takes ~0.5-5 s per object, so the training env samples from a precomputed library:

    python -m aaltonomous.rl.scripted --out aaltonomous/rl/data/lib_left.npz --workers 48

Library entry k: shape, true object pose (pelvis xy, yaw), the start state (arm (14,), hand (14,) at the end of
`to_pregrasp`) and a fixed-length 30 Hz target trajectory arm (T, 14), hand (T, 14) padded with the last target,
plus the tick index where each phase starts. The env places the object at the entry's pose + placement noise,
so the script was planned for a slightly wrong pose, which is what happens with perception too.
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import time

import numpy as np

SHAPES = ("cube", "cylinder", "triangle")
HZ = 30.0
DT = 1.0 / HZ
# Tail appended after `lift` (seconds): hold, shake, hold. The success check (env) needs the object >= 10 cm up for
# HOLD_S, still up after the shake.
HOLD_S = 1.0
SHAKE_S = 1.0
POST_S = 0.5
SHAKE_HZ = 2.0
# Shake = sinusoids on left shoulder pitch / shoulder roll / wrist yaw (arm idx 0, 1, 6), amplitude in rad.
# Peak joint speed = 2 pi f A = 0.75 rad/s for A = 0.06 (cap 1.0).
SHAKE_AMP = {0: 0.06, 1: 0.04, 6: 0.06}
PHASES = ("descend", "close", "lift", "hold", "shake", "post")
# Left arm region (pelvis frame): the object band x 0.26-0.33, the left half of y plus 2 cm across the centre line.
REGION_X = (0.26, 0.33)
REGION_Y = (-0.02, 0.17)
YAW_MAT_DEG = (-45.0, 45.0)


def make_planner():
    """The oracle's planner without the scene client (privileged poses are passed in directly)."""
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    from aaltonomous.sim.oracle import ORACLE_OVERRIDES, _OraclePlanner
    base = ClassicalPolicy(overrides=dict(ORACLE_OVERRIDES), threaded=False, log=None)
    pl = _OraclePlanner(base.cfg, base.scene, base.post, base.home, base.kin, base.safety, base.spec,
                        mapping=dict(base.mapping))
    pl.drop_offsets = {}
    return pl, base


def mat_yaw_to_pelvis(yaw_mat):
    from aaltonomous.sim.aalto_task.layout import mat_yaw_to_pelvis as f
    return f(yaw_mat)


def plan_left(pl, base, shape, x, y, yaw_pelvis, bin_name="left", from_home=False):
    """Left-arm plan for an object resting at pelvis (x, y) with pelvis yaw. Returns dict or None."""
    from aaltonomous.baselines.classical.perception import Detection
    sc = base.scene
    H = sc.object_height(shape)
    top = np.array([x, y, sc.counter_z + H])
    d = Detection("red", top, 1000, 1000, "oracle", yaw=float(yaw_pelvis))
    d.shape, d.name = shape, "obj"
    arm0, hand0 = base.home.copy(), base.post["open"].copy()
    p = pl._plan_side_hints(d, "left", bin_name, arm0, hand0)
    if not p.ok:
        return {"ok": False, "reason": p.reason}
    ph = {n: (s, e) for n, s, e in p.phases}
    s0 = 0 if from_home else ph["to_pregrasp"][1]  # first tick after reaching the pregrasp (or the whole plan)
    e_lift = ph["lift"][1]
    arm, hand = p.arm[s0:e_lift], p.hand[s0:e_lift]
    if from_home:
        start_arm, start_hand = arm0.copy(), hand0.copy()
    else:
        start_arm, start_hand = p.arm[s0 - 1].copy(), p.hand[s0 - 1].copy()
    starts = {"descend": 0, "close": ph["close"][0] - s0, "lift": ph["lift"][0] - s0}
    n_hold, n_shake, n_post = int(round(HOLD_S * HZ)), int(round(SHAKE_S * HZ)), int(round(POST_S * HZ))
    last_a, last_h = arm[-1], hand[-1]
    t = np.arange(1, n_shake + 1) * DT
    shake = np.repeat(last_a[None], n_shake, 0)
    for j, A in SHAKE_AMP.items():
        shake[:, j] += A * np.sin(2 * np.pi * SHAKE_HZ * t)
    tail_a = np.vstack([np.repeat(last_a[None], n_hold, 0), shake, np.repeat(last_a[None], n_post, 0)])
    tail_h = np.repeat(last_h[None], len(tail_a), 0)
    starts["hold"] = len(arm)
    starts["shake"] = len(arm) + n_hold
    starts["post"] = len(arm) + n_hold + n_shake
    return {"ok": True, "arm": np.vstack([arm, tail_a]), "hand": np.vstack([hand, tail_h]),
            "start_arm": start_arm, "start_hand": start_hand, "starts": starts, "reason": p.reason,
            "face": p.reason}


_W = None


def _worker_init():
    global _W
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W = make_planner()


def _job(args):
    k, shape, x, y, yaw_mat = args
    pl, base = _W
    t0 = time.monotonic()
    try:
        r = plan_left(pl, base, shape, x, y, mat_yaw_to_pelvis(np.radians(yaw_mat)))
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "reason": repr(e)}
    r.update(k=k, shape=shape, x=x, y=y, yaw_mat=yaw_mat, plan_s=time.monotonic() - t0)
    return r


def build(nx=4, ny=6, nyaw=10, workers=32, out=None, shapes=SHAPES, seed=0):
    xs = np.linspace(*REGION_X, nx)
    ys = np.linspace(*REGION_Y, ny)
    yaws = np.linspace(*YAW_MAT_DEG, nyaw)
    jobs = [(k, s, float(x), float(y), float(w))
            for k, (s, x, y, w) in enumerate(itertools.product(shapes, xs, ys, yaws))]
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(workers, initializer=_worker_init) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} planned, {time.monotonic() - t0:.0f} s", flush=True)
    res.sort(key=lambda r: r["k"])
    ok = [r for r in res if r["ok"]]
    print(f"{len(ok)}/{len(res)} plans ok; failures by shape:",
          {s: sum(1 for r in res if r["shape"] == s and not r["ok"]) for s in shapes})
    T = max(len(r["arm"]) for r in ok)
    n = len(ok)
    arm = np.zeros((n, T, 14), np.float32)
    hand = np.zeros((n, T, 14), np.float32)
    length = np.zeros(n, np.int32)
    starts = np.zeros((n, len(PHASES)), np.int32)
    for i, r in enumerate(ok):
        L = len(r["arm"])
        arm[i, :L], hand[i, :L] = r["arm"], r["hand"]
        arm[i, L:], hand[i, L:] = r["arm"][-1], r["hand"][-1]
        length[i] = L
        starts[i] = [r["starts"][p] for p in PHASES]
    lib = dict(
        shape=np.array([SHAPES.index(r["shape"]) for r in ok], np.int32),
        x=np.array([r["x"] for r in ok], np.float32), y=np.array([r["y"] for r in ok], np.float32),
        yaw_mat=np.radians(np.array([r["yaw_mat"] for r in ok], np.float32)),
        yaw_pelvis=np.array([mat_yaw_to_pelvis(np.radians(r["yaw_mat"])) for r in ok], np.float32),
        start_arm=np.array([r["start_arm"] for r in ok], np.float32),
        start_hand=np.array([r["start_hand"] for r in ok], np.float32),
        arm=arm, hand=hand, length=length, starts=starts, phases=np.array(PHASES),
        failed=np.array([f"{r['shape']} {r['x']:.3f} {r['y']:.3f} {r['yaw_mat']:.0f}: {r['reason']}"
                         for r in res if not r["ok"]]),
    )
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, **lib)
        print(f"wrote {out}: {n} entries, T={T} ticks ({T / HZ:.1f} s)")
    return lib


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="aaltonomous/rl/data/lib_left.npz")
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--nyaw", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    a = ap.parse_args(argv)
    build(a.nx, a.ny, a.nyaw, a.workers, a.out, tuple(a.shapes.split(",")))


if __name__ == "__main__":
    main()
