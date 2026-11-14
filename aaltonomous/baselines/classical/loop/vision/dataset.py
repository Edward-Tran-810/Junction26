"""Frame sets for the vision loop: named, frozen splits of synthetic frames (and, later, recorded sim/real frames).

    python -m aaltonomous.baselines.classical.loop.vision.dataset --split dev --render     # fill the cache
    python -m aaltonomous.baselines.classical.loop.vision.dataset --split fresh_01 --freeze  # write the manifest

A split is a list of frame ids. A SYNTHETIC frame id is "<family>-<layout_mode>-<seed>" (synth.py); the frame is
re-generated from it bit-exactly, so only the manifest (ids + SHA-256 of the pixels) is committed and the pixels
live in a gitignored cache (loop/vision/cache/). A RECORDED frame id is "rec:<path to .npz>" (head_rgb, head_depth
and a json sidecar with the ground truth, see capture_sim.py); those files are never regenerated. The recorded
Isaac set lives in loop/vision/recorded/isaac_r0/ (gitignored; $AALTO_VISION_RECORDED overrides) and is split by
seed order into isaac_dev (120), isaac_fresh_01..06 (40 each) and isaac_holdout (the rest).

Splits and their roles (never mix them; never tune on anything but `dev`):
    dev        tuning set. Free to look at and tune on. family train + nominal, band + wide layouts.
    fresh_NN   one per round, measured once BEFORE any change of that round, then absorbed into regression.
               Seeds 1000*NN + i: never used by any earlier split. Mix: train + unseen families.
    holdout    final synthetic holdout, seeds >= 900000. Measured once at the end of the offline work, never tuned on.
Disjoint seed ranges + one frame per (seed, family, mode) = no two frames share a layout or noise draw.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # shared machine: one core per worker
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MANIFESTS = HERE / "manifests"


RECORDED = Path(os.environ.get("AALTO_VISION_RECORDED", str(HERE / "recorded" / "isaac_r0")))
# recorded Isaac set (capture_sim.py), ordered by seed: dev 120 | fresh_01..06 40 each | holdout = the rest (>= 60)
ISAAC_DEV, ISAAC_FRESH, ISAAC_N_FRESH = 120, 40, 6


def recorded_ids(root: Path | None = None) -> list[str]:
    """'rec:<path>.npz' ids of a recorded set (default: RECORDED), sorted by seed (index.jsonl of capture_sim.py)."""
    root = Path(root) if root is not None else RECORDED
    idx = root / "index.jsonl"
    if not idx.exists():
        raise FileNotFoundError(f"no recorded frames at {root} (vision round 0 not captured yet, see capture_sim.py)")
    rows = sorted((json.loads(line) for line in idx.read_text().splitlines() if line.strip()), key=lambda r: r["seed"])
    return [f"rec:{root / (r['id'] + '.npz')}" for r in rows]


def split_ids(name: str) -> list[str]:
    """Frame ids of a named split (deterministic; see the module docstring for the roles).
    isaac_dev / isaac_fresh_NN (01-06) / isaac_holdout: the recorded Isaac round-0 set (not captured yet)."""
    if name.startswith("isaac_"):
        ids = recorded_ids()
        if name == "isaac_dev":
            return ids[:ISAAC_DEV]
        if name.startswith("isaac_fresh_"):
            k = int(name.rsplit("_", 1)[1])
            if not 1 <= k <= ISAAC_N_FRESH:
                raise ValueError(f"isaac fresh slices are 01..{ISAAC_N_FRESH:02d}")
            b = ISAAC_DEV + (k - 1) * ISAAC_FRESH
            return ids[b:b + ISAAC_FRESH]
        if name == "isaac_holdout":
            return ids[ISAAC_DEV + ISAAC_N_FRESH * ISAAC_FRESH:]
        raise ValueError(f"unknown split {name!r}")
    if name == "dev":
        return ([f"train-band-{s}" for s in range(0, 80)] + [f"train-wide-{s}" for s in range(80, 160)]
                + [f"nominal-band-{s}" for s in range(160, 200)])
    if name.startswith("fresh_"):
        k = int(name.split("_")[1])
        b = 1000 * k
        return ([f"train-band-{b + i}" for i in range(0, 40)] + [f"unseen-band-{b + i}" for i in range(40, 80)]
                + [f"unseen-wide-{b + i}" for i in range(80, 120)])
    if name == "holdout":
        b = 900000
        return ([f"train-band-{b + i}" for i in range(0, 80)] + [f"unseen-band-{b + i}" for i in range(80, 160)]
                + [f"unseen-wide-{b + i}" for i in range(160, 240)])
    if name.startswith("regression"):          # every fresh slice measured so far: regression_NN = fresh_01..NN
        k = int(name.split("_")[1])
        return [i for j in range(1, k + 1) for i in split_ids(f"fresh_{j:02d}")]
    # ---- PERCEPTION_MILESTONES A2 / A3 (synth_extra.py). dev = tuning; fresh_NN measured once, then regression.
    #      Seed ranges disjoint from every split above (2_000_000+).
    for kind, b in (("close", 2_000_000), ("occl", 2_100_000)):          # A2: 60 frames each, train + unseen base
        if name == f"a2_{kind}_dev":
            return [f"train-{kind}-{b + i}" for i in range(30)] + [f"unseen-{kind}-{b + i}" for i in range(30, 60)]
        if name.startswith(f"a2_{kind}_fresh_"):
            k = int(name.rsplit("_", 1)[1])
            o = b + 1000 * k
            return [f"train-{kind}-{o + i}" for i in range(30)] + [f"unseen-{kind}-{o + i}" for i in range(30, 60)]
    if name == "shift_dev":                                              # A3 harder family
        return [f"shift-band-{2_200_000 + i}" for i in range(40)] + [f"shift-wide-{2_200_000 + i}" for i in range(40, 80)]
    if name.startswith("shift_fresh_"):
        o = 2_200_000 + 1000 * int(name.rsplit("_", 1)[1])
        return [f"shift-band-{o + i}" for i in range(40)] + [f"shift-wide-{o + i}" for i in range(40, 80)]
    if name.startswith("pose_"):                                         # A4 camera pose error: pose_<k>
        k = name.split("_", 1)[1]
        return [f"pose{k}-band-{2_300_000 + i}" for i in range(30)]
    if name.startswith("smoke"):               # tiny set for unit tests
        return ["nominal-band-1", "train-band-2", "unseen-band-3", "unseen-wide-4"]
    raise ValueError(f"unknown split {name!r}")


def parse_id(fid: str):
    fam, mode, seed = fid.split("-")
    return fam, mode, int(seed)


_RENDERER = None


def synth_hash() -> str:
    """First 12 hex of the SHA-256 of synth.py: a frame cache is only valid for the generator that made it."""
    import hashlib
    return hashlib.sha256((HERE / "synth.py").read_bytes()).hexdigest()[:12]


def cache_dir(fid: str | None = None) -> Path:
    """Frame cache of the current generator. synth_extra frames (A2 / A3) are keyed by synth.py + synth_extra.py."""
    if fid is not None and _is_extra(fid):
        import hashlib
        h = hashlib.sha256((HERE / "synth.py").read_bytes() + (HERE / "synth_extra.py").read_bytes()).hexdigest()
        return CACHE / f"x{h[:12]}"
    return CACHE / synth_hash()


def _is_extra(fid: str) -> bool:
    if fid.startswith("rec:"):
        return False
    from .synth_extra import is_extra
    fam, mode, _ = parse_id(fid)
    return is_extra(fam, mode)


def _renderer():
    global _RENDERER
    if _RENDERER is None:
        from .synth import SynthRenderer
        _RENDERER = SynthRenderer()
    return _RENDERER


def load_frame(fid: str, use_cache: bool = True):
    """(rgb, depth, gt) of one frame id. Synthetic frames are rendered on first use and cached as .npz + .json."""
    if fid.startswith("rec:"):
        p = Path(fid[4:])
        z = np.load(p)
        gt = json.loads(p.with_suffix(".json").read_text())
        return z["head_rgb"], z["head_depth"], gt
    path = cache_dir(fid) / f"{fid}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return z["rgb"], z["depth"], json.loads(path.with_suffix(".json").read_text())
    from .synth import sample_spec
    fam, mode, seed = parse_id(fid)
    if _is_extra(fid):
        from .synth_extra import sample_spec_extra
        spec = sample_spec_extra(seed, fam, mode)
    else:
        spec = sample_spec(seed, fam, mode)
    rgb, depth, gt = _renderer().render(spec)
    for k in ("close_pair", "close_dist", "occl_target", "occl_frac_est", "shift_style"):
        if k in spec:
            gt[k] = spec[k]
    gt["id"] = fid
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=rgb, depth=depth)
        path.with_suffix(".json").write_text(json.dumps(gt))
    return rgb, depth, gt


def _render_one(fid):
    rgb, depth, gt = load_frame(fid)
    from .synth import frame_hash
    return fid, frame_hash(rgb, depth), len(gt["objects"])


def render_split(name: str, workers: int = 8):
    """Render (or load) every frame of a split in parallel. Returns {id: hash} (recorded frames: their hash)."""
    ids = split_ids(name)
    if all(i.startswith("rec:") for i in ids):
        from .synth import frame_hash
        return {i: frame_hash(*load_frame(i)[:2]) for i in ids}
    todo = [i for i in ids if not (cache_dir(i) / f"{i}.npz").exists()]
    out = {}
    if todo:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for fid, h, n in pool.imap_unordered(_render_one, todo, chunksize=2):
                out[fid] = h
    for fid in ids:
        if fid not in out:
            out[fid] = _render_one(fid)[1]
    return {i: out[i] for i in ids}


def freeze(name: str, workers: int = 8) -> Path:
    """Write manifests/<name>.json: ids, frame hashes, generator source hash. Commit it BEFORE measuring."""
    import hashlib
    hashes = render_split(name, workers)
    src = hashlib.sha256((HERE / "synth.py").read_bytes()).hexdigest()[:16]
    extra = any(_is_extra(i) for i in hashes)
    xsrc = hashlib.sha256((HERE / "synth_extra.py").read_bytes()).hexdigest()[:16] if extra else None
    MANIFESTS.mkdir(exist_ok=True)
    p = MANIFESTS / f"{name}.json"
    p.write_text(json.dumps({"split": name, "n_frames": len(hashes), "generator": "synth.py" + (" + synth_extra.py" if extra else ""), "synth_sha256": src,
                             **({"synth_extra_sha256": xsrc} if extra else {}),
                             "frames": hashes}, indent=0))
    return p


def verify(name: str) -> list[str]:
    """Frame ids whose re-generated pixels differ from the frozen manifest (empty list = reproducible)."""
    m = json.loads((MANIFESTS / f"{name}.json").read_text())
    from .synth import frame_hash
    bad = []
    for fid, h in m["frames"].items():
        rgb, depth, _ = load_frame(fid, use_cache=False)
        if frame_hash(rgb, depth) != h:
            bad.append(fid)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--render", action="store_true", help="render the split into the cache")
    ap.add_argument("--freeze", action="store_true", help="render + write the manifest")
    ap.add_argument("--verify", action="store_true", help="re-render without the cache and compare hashes")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.freeze:
        print("manifest ->", freeze(a.split, a.workers))
    elif a.render:
        h = render_split(a.split, a.workers)
        print(f"{len(h)} frames in {cache_dir(next(iter(h)))}")
    if a.verify:
        bad = verify(a.split)
        print("verify:", "OK (all frames bit-exact)" if not bad else f"{len(bad)} differ: {bad[:5]}")


if __name__ == "__main__":
    main()
