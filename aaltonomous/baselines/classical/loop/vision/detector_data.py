"""PERCEPTION_MILESTONES A3 `detector`: YOLO-seg training data from the synthetic renderer (instance masks from the
ray caster's per-pixel solid ids). Seeds 3_000_000+ (train) and 3_500_000+ (val): disjoint from every evaluation
split (dev / fresh / holdout / a2 / shift / pose), so detector scores on those splits are not on training frames.

    python -m aaltonomous.baselines.classical.loop.vision.detector_data --out <dir> --n-train 2400 --n-val 300

Mix per frame (drawn from the seed): family train / unseen / shift, layout band / wide / close / occl.
Classes: 9 = colour x shape, index = 3 * colour_index + shape_index (tasks.COLORS x scene.SHAPES order).
Objects with < 60 visible pixels are not labelled (the detector should not learn to hallucinate them).
SYNTHETIC: the detector only ever saw rendered frames.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aaltonomous.interface import tasks  # noqa: E402
from aaltonomous.scene.scene import SHAPES  # noqa: E402

from . import synth  # noqa: E402

CLASSES = [f"{c}_{s}" for c in tasks.COLORS for s in SHAPES]


class MaskRenderer(synth.SynthRenderer):
    def render_masks(self, spec):
        """(rgb, depth, gt, instance map int16 (-1 = no object, j = spec object j))."""
        rgb, depth, gt = self.render(spec)
        R, o = self._camera(spec)
        solids = self.solids(spec)
        _, kind, sid, _, _ = self._cast(R, o, solids, 0.0, 0.0)
        inst = np.full(kind.shape, -1, np.int16)
        for j in range(len(spec["objects"])):
            k = next(i for i, s in enumerate(solids) if s["tag"] == f"obj:{j}")
            inst[(kind == synth.KIND_SOLID) & (sid == k)] = j
        return rgb, depth, gt, inst


def spec_for(seed: int):
    from .synth_extra import sample_spec_extra
    rng = np.random.default_rng([seed, 777])
    fam = str(rng.choice(["train", "unseen", "shift"], p=[0.35, 0.35, 0.3]))
    mode = str(rng.choice(["band", "wide", "close", "occl"], p=[0.35, 0.35, 0.15, 0.15]))
    if fam == "shift":
        return sample_spec_extra(seed, "shift", mode if mode in synth.LAYOUT_MODES else "wide")
    if mode in ("close", "occl"):
        return sample_spec_extra(seed, fam, mode)
    return synth.sample_spec(seed, fam, mode)


_R = None


def _one(args):
    import cv2
    seed, out, split = args
    global _R
    if _R is None:
        _R = MaskRenderer()
    try:
        spec = spec_for(seed)
    except RuntimeError:
        return 0
    rgb, _, _, inst = _R.render_masks(spec)
    H, W = inst.shape
    lines = []
    for j, ob in enumerate(spec["objects"]):
        m = (inst == j).astype(np.uint8)
        if m.sum() < 60:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        if len(c) < 3:
            continue
        cls = 3 * tasks.COLORS.index(ob["color"]) + SHAPES.index(ob.get("shape", "cube"))
        lines.append(f"{cls} " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c))
    name = f"s{seed}"
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    out = Path(a.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    jobs = [(3_000_000 + i, out, "train") for i in range(a.n_train)] + \
           [(3_500_000 + i, out, "val") for i in range(a.n_val)]
    jobs = [j for j in jobs if not (out / "labels" / j[2] / f"s{j[0]}.txt").exists()]
    with mp.get_context("fork").Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(_one, jobs, chunksize=4))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
                                   + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"{len(jobs)} frames rendered, {n} instances -> {out}")


if __name__ == "__main__":
    main()
