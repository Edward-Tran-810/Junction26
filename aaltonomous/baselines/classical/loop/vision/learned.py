"""PERCEPTION_MILESTONES A3 learned candidates, plugged into perception v2 behind the same Detection output.

    perception.segmenter: geometry (default, v2) | detector | open_vocab
        detector     YOLO-seg trained on synthetic frames only (detector_data.py); 9 classes colour x shape
        open_vocab   OWLv2 zero-shot boxes for 9 text queries ("a red cube", ...), no training
      Each yields (colour, shape hint, score, full-image bool mask). Perception keeps the depth part of v2: the mask
      is intersected with object-height depth pixels, then the same top-face polygon fit gives position, yaw and
      the shape. The model's shape is used only when the fit cannot decide (e.g. no depth).
    perception.vlm_verify: true
        v2 as is, then a vision-language model (Qwen2.5-VL-3B-Instruct) answers colour + shape on a crop around each
        detection v2 is unsure about (shape None or colour vote < vlm_below_conf). A missing shape is filled from
        the VLM; a colour the VLM disagrees with becomes None (abstain), never the VLM's colour.

Heavy imports (torch, ultralytics, transformers) are lazy: this module is only imported when a learned option is on.
GPU device: $AALTO_LEARNED_DEVICE (default cuda:0).
"""
from __future__ import annotations

import os
import re

import numpy as np

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
DEV = os.environ.get("AALTO_LEARNED_DEVICE", "cuda:0")
_CACHE = {}


def _yolo(weights):
    if ("yolo", weights) not in _CACHE:
        from ultralytics import YOLO
        _CACHE[("yolo", weights)] = YOLO(weights)
    return _CACHE[("yolo", weights)]


def detector_regions(rgb, weights, conf=0.35):
    """[(colour, shape, score, mask HxW bool)] from the YOLO-seg detector."""
    m = _yolo(weights)
    r = m.predict(rgb[:, :, ::-1].copy(), conf=conf, device=DEV, verbose=False, retina_masks=True)[0]
    out = []
    if r.masks is None:
        return out
    masks = r.masks.data.cpu().numpy() > 0.5
    for k, (c, s) in enumerate(zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())):
        out.append((COLORS[c // 3], SHAPES[c % 3], float(s), masks[k]))
    return out


def queries(words=None):
    """9 text queries; `words` maps the internal colour names to the words used for the real objects
    (e.g. {"red": "orange", "blue": "purple"}), class index = 3 * colour + shape either way."""
    w = words or {}
    return [f"a {w.get(c, c)} {s}" for c in COLORS for s in ("cube", "cylinder", "triangular prism")]


def _owl():
    if "owl" not in _CACHE:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        name = "google/owlv2-base-patch16-ensemble"
        proc = Owlv2Processor.from_pretrained(name)
        model = Owlv2ForObjectDetection.from_pretrained(name, dtype=torch.float16).to(DEV).eval()
        _CACHE["owl"] = (proc, model)
    return _CACHE["owl"]


def open_vocab_regions(rgb, thr=0.15, nms_iou=0.4, words=None):
    """[(colour, shape, score, box mask HxW bool)] from OWLv2 zero-shot boxes (class-agnostic NMS)."""
    import torch
    from PIL import Image
    proc, model = _owl()
    H, W = rgb.shape[:2]
    img = Image.fromarray(rgb)
    inp = proc(text=[queries(words)], images=img, return_tensors="pt").to(DEV)
    inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    S = max(H, W)                                    # OWLv2 pads to a square: boxes are relative to it
    res = proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=[(S, S)])[0]
    boxes, scores, labels = res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), \
        res["labels"].cpu().numpy()
    order = np.argsort(-scores)
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < nms_iou for j in keep):
            keep.append(i)
    regs = []
    for i in keep:
        x0, y0, x1, y1 = np.clip(boxes[i], 0, [W, H, W, H]).astype(int)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        c, s = divmod(int(labels[i]), 3)
        regs.append((COLORS[c], SHAPES[s], float(scores[i]), m))
    return regs


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


def _vlm():
    if "vlm" not in _CACHE:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        name = os.environ.get("AALTO_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
        _CACHE["vlm"] = (proc, model)
    return _CACHE["vlm"]


VLM_PROMPT = ("This crop shows one toy block on a table (ignore the table, boxes and robot hands). "
              "Answer with exactly two words: its colour (red, yellow or blue) and its shape (cube, cylinder or "
              "triangle). If you cannot tell, answer 'unknown unknown'.")


def vlm_name(crop_rgb):
    """(colour or None, shape or None) for one crop."""
    import torch
    from PIL import Image
    proc, model = _vlm()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[Image.fromarray(crop_rgb)], return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].lower()
    words = re.findall(r"[a-z]+", ans)
    col = next((w for w in words if w in COLORS), None)
    shp = next((w for w in words if w in SHAPES or w in ("prism", "cylindrical")), None)
    shp = {"prism": "triangle", "cylindrical": "cylinder"}.get(shp, shp)
    return col, shp


def crop(rgb, uv, half=56):
    H, W = rgb.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    return rgb[max(0, v - half):min(H, v + half), max(0, u - half):min(W, u + half)]


# ------------------------------------------------------------------ detector + hue colour hybrid
def region_hue(hsv, mask, min_s=60, v_lo=40, v_hi=250):
    """Saturation-weighted circular mean hue (OpenCV 0-180) of a region's well-exposed coloured pixels, or None.
    Glare (V > v_hi), shadow (V < v_lo) and grey pixels (S < min_s) carry no hue evidence."""
    p = hsv[mask]
    p = p[(p[:, 1] >= min_s) & (p[:, 2] >= v_lo) & (p[:, 2] <= v_hi)]
    if len(p) < 15:
        return None
    a = p[:, 0].astype(np.float64) * (2 * np.pi / 180)
    w = p[:, 1].astype(np.float64)
    h = np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()) * 180 / (2 * np.pi)
    return float(h % 180)


def hdist(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def assign_colours(hues, protos, gap=10.0, max_dist=25.0):
    """Joint per-frame colour assignment. hues: [hue or None] per region; protos: {colour: hue}.
    1. 1-D circular clustering of the region hues (a new cluster where neighbours differ by > gap), at most
       len(protos) clusters (closest clusters merged).
    2. Clusters -> colours by minimum total circular distance (Hungarian), each colour used at most once: the
       hue ORDER of the objects in the frame decides, so a global cast or a different object palette (orange
       instead of red) keeps the classes apart where independent nearest-prototype labels would collide.
    3. A region farther than max_dist from its colour's prototype is None (abstain).
    Returns [(colour or None, distance)]."""
    from scipy.optimize import linear_sum_assignment
    idx = [i for i, h in enumerate(hues) if h is not None]
    out = [(None, None)] * len(hues)
    if not idx:
        return out
    order = sorted(idx, key=lambda i: hues[i])
    clusters = [[order[0]]]
    for i in order[1:]:
        if hues[i] - hues[clusters[-1][-1]] > gap:
            clusters.append([i])
        else:
            clusters[-1].append(i)
    if len(clusters) > 1 and hdist(hues[clusters[0][0]], hues[clusters[-1][-1]]) <= gap:   # wrap around 180
        clusters[0] = clusters.pop() + clusters[0]

    def centre(c):
        a = np.array([hues[i] for i in c]) * (2 * np.pi / 180)
        return float(np.arctan2(np.sin(a).sum(), np.cos(a).sum()) * 180 / (2 * np.pi) % 180)
    while len(clusters) > len(protos):
        cs = [centre(c) for c in clusters]
        k = min(range(len(cs)), key=lambda k: hdist(cs[k], cs[(k + 1) % len(cs)]))
        j = (k + 1) % len(clusters)
        clusters[k] = clusters[k] + clusters[j]
        clusters.pop(j)
    names = list(protos)
    cs = [centre(c) for c in clusters]
    C = np.array([[hdist(c, protos[n]) for n in names] for c in cs])
    r, col = linear_sum_assignment(C)
    for ci, pi in zip(r, col):
        for i in clusters[ci]:
            d = hdist(hues[i], protos[names[pi]])
            out[i] = (names[pi] if d <= max_dist else None, d)
    return out
