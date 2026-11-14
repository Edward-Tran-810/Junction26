"""Library for an explicit list of cells (e.g. the DDS grasp_test grid), left arm. Right-arm cells are mirrored onto the
left arm (pelvis y -> -y, pelvis yaw -> -yaw; the shapes are mirror-symmetric), see skill.py for the joint mirror.

    python -m aaltonomous.rl.cells --grasp-test --out aaltonomous/rl/data/lib_grasptest.npz
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp

import numpy as np

from . import scripted as S


def grasp_test_cells(yaws=(-45, -20, 0, 20, 45)):
    """sim/grasp_test.py default grid (mat x 4 points over region_x, mat y middle) with the arm the stock oracle
    uses for the colour grasp_test cycles to (red -> left bin -> left arm, blue -> right, yellow -> object side)."""
    from aaltonomous.scene.scene import Scene
    sc = Scene.load()
    L = sc.cfg["layout"]
    xs = list(np.linspace(*L["region_x"], 4))
    ys = [float(np.mean(L["region_y"]))]
    colors = itertools.cycle(["red", "yellow", "blue"])
    out = []
    for shape, yaw, x, y in itertools.product(S.SHAPES, yaws, xs, ys):
        c = next(colors)
        p = sc.mat_to_pelvis([x, y, 0.0])
        arm = "left" if c == "red" or (c == "yellow" and p[1] > 0) else "right"
        out.append(dict(shape=shape, yaw_mat=float(yaw), x_mat=float(x), y_mat=float(y), color=c, side=arm,
                        px=float(p[0]), py=float(p[1])))
    return out


def _job(c):
    pl, base = S._W
    yaw_p = S.mat_yaw_to_pelvis(np.radians(c["yaw_mat"]))
    y = c["py"]
    if c["side"] == "right":
        y, yaw_p = -y, -yaw_p
    b = {"red": "left", "yellow": "center", "blue": "right"}[c["color"]] if c.get("real_bin", True) else "left"
    if c["side"] == "right":
        b = {"left": "right", "right": "left", "center": "center"}[b]     # mirrored task
    r = S.plan_left(pl, base, c["shape"], c["px"], y, yaw_p, b, from_home=c.get("from_home", False))
    r.update(c, x=c["px"], y=y, yaw_pelvis=yaw_p)
    return r


def build(cells, out, workers=16):
    with mp.get_context("fork").Pool(workers, initializer=S._worker_init) as pool:
        res = pool.map(_job, cells)
    ok = [r for r in res if r["ok"]]
    print(f"{len(ok)}/{len(res)} planned;", [f"{r["shape"]} {r["side"]} {r["py"]:+.3f} {r['yaw_mat']}: {r['reason']}"
                                            for r in res if not r["ok"]])
    T = max(len(r["arm"]) for r in ok)
    n = len(ok)
    arm = np.zeros((n, T, 14), np.float32); hand = np.zeros((n, T, 14), np.float32)
    for i, r in enumerate(ok):
        L = len(r["arm"])
        arm[i, :L], hand[i, :L] = r["arm"], r["hand"]
        arm[i, L:], hand[i, L:] = r["arm"][-1], r["hand"][-1]
    np.savez_compressed(out,
        shape=np.array([S.SHAPES.index(r["shape"]) for r in ok], np.int32),
        x=np.array([r["x"] for r in ok], np.float32), y=np.array([r["y"] for r in ok], np.float32),
        yaw_mat=np.radians(np.array([r["yaw_mat"] for r in ok], np.float32)),
        yaw_pelvis=np.array([r["yaw_pelvis"] for r in ok], np.float32),
        start_arm=np.array([r["start_arm"] for r in ok], np.float32),
        start_hand=np.array([r["start_hand"] for r in ok], np.float32),
        arm=arm, hand=hand, length=np.array([len(r["arm"]) for r in ok], np.int32),
        starts=np.array([[r["starts"][p] for p in S.PHASES] for r in ok], np.int32), phases=np.array(S.PHASES),
        cell_arm=np.array([r["side"] for r in ok]), cell_color=np.array([r["color"] for r in ok]),
        cell_ymat=np.array([r["x_mat"] for r in ok], np.float32), face=np.array([r["face"] for r in ok]),
        failed=np.array([]))
    print("wrote", out, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="aaltonomous/rl/data/lib_grasptest.npz")
    ap.add_argument("--from-home", action="store_true", help="start at the home pose (DDS start), not the pregrasp")
    a = ap.parse_args()
    cells = grasp_test_cells()
    for c in cells:
        c["from_home"] = a.from_home
    build(cells, a.out)
