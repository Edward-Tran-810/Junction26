"""Side-by-side video of perception approaches on real head frames (one tile per approach, same frame).

    python -m aaltonomous.baselines.classical.loop.vision.render_video \
        --root /workspace/data/real_drops/frames/g1_junction_drop_2 --out drop2_approaches.mp4

Per tile: every Detection at its blob pixel (uv): filled dot + ring in the detected colour (grey = colour None),
label "<colour> <shape>" ("?" = shape None). Labelled frames (real/drop2_labels.json) also show the hand labels
as white crosses (not drawn as the approach's output). Footer: frame id + the episode's task text.
H.264 via the ffmpeg binary bundled with imageio-ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .eval_vision import make_perception

HERE = Path(__file__).resolve().parent
BGR = {"red": (40, 40, 230), "yellow": (0, 220, 255), "blue": (230, 90, 30), None: (170, 170, 170)}
DEFAULT = ["legacy", "v2", "open_vocab", "vlm_verify", "detector", "detector_hue_real"]


def tile(rgb, dets, name, labels, scale):
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for l in labels or []:
        u, v = int(l[2]), int(l[3])
        cv2.drawMarker(im, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    for d in dets:
        u, v = int(round(d.uv[0])), int(round(d.uv[1]))
        c = BGR.get(d.color, BGR[None])
        cv2.circle(im, (u, v), 7, c, -1)
        cv2.circle(im, (u, v), 11, (0, 0, 0), 2)
        txt = f"{d.color or 'unk'} {d.shape or '?'}"
        cv2.putText(im, txt, (u + 12, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(im, txt, (u + 12, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, f"{name}   {len(dets)} det", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return im


def main(argv=None):
    import imageio_ffmpeg
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/data/real_drops/frames/g1_junction_drop_2")
    ap.add_argument("--variant", nargs="+", default=DEFAULT)
    ap.add_argument("--labels", default=str(HERE / "real" / "drop2_labels.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.75)
    a = ap.parse_args(argv)
    root = Path(a.root)
    index = json.loads((root / "index.json").read_text())
    labels = json.loads(Path(a.labels).read_text())["frames"] if Path(a.labels).exists() else {}
    pers = {v: make_perception(v) for v in a.variant}
    tmp = Path(tempfile.mkdtemp(dir=str(Path(a.out).resolve().parent)))
    for k, r in enumerate(index):
        z = np.load(root / f"{r['id']}.npz")
        rgb, depth = z["rgb"], z["depth"]
        tiles = []
        for v, per in pers.items():
            per.reset_calibration()
            tiles.append(tile(rgb, per.detect(rgb, depth), v, labels.get(r["id"]), a.scale))
        while len(tiles) % a.cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
        frame = np.vstack(rows)
        foot = np.zeros((34, frame.shape[1], 3), np.uint8)
        lab = "  [hand-labelled: white crosses]" if r["id"] in labels else ""
        cv2.putText(foot, f"{root.name}  {r['id']}  t={r['t']}s  task: {r['task']}{lab}   (REAL, no depth)",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(tmp / f"f{k:05d}.png"), np.vstack([frame, foot]))
        if k % 20 == 0:
            print(f"  {k}/{len(index)} frames")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out], check=True)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    print("video ->", a.out)


if __name__ == "__main__":
    main()
