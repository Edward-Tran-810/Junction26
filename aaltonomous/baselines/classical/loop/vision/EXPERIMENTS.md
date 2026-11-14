# Perception and language experiments: comparison log

One row per experiment, failed approaches included. Plan: `/workspace/context/PERCEPTION_MILESTONES.md` (adopted by
manh 2026-11-14). Round-by-round diary: `loop/STATE.md`.

**Evidence labels** (every number carries one): `synthetic` (loop/vision/synth.py), `mock world`
(mock_world.py), `Isaac nominal`, `Isaac DR`, `Isaac held-out-style`, `real`. A synthetic number is never a
stop-rule result.

**Scoring reminder:** `sort_language` +2 correct bin, -1 wrong bin, 0 left alone. So `wrong` (-1) and `unknown` (0)
are always counted separately.

Python used for everything below: personal venv `~/venvs/lv` (uv, Python 3.12, numpy 2.2.6, scipy 1.15.2,
opencv-headless 4.10.0.84, pin 3.3.1: the same pins as `docker/baseline.Dockerfile`). Machine: shared 60-core
x86_64, 2x RTX PRO 6000 Blackwell (GPU 0 for these experiments; GPU 1 is long's RL training).

## A0. Submission limits (what an approach may use)

| Limit | Value | Source |
|---|---|---|
| Hardware at finals | One PC with **RTX 5090, 32 GB VRAM**, wired to the robot | `docs/participants/FAQ.md` ("What runs the finals?"), `SUBMISSION.md` §1 |
| GPU | Allowed. `docker run -d --gpus all`; must fit in 32 GB VRAM | `SUBMISSION.md` §1 |
| Base image | linux/amd64, CUDA >= 12.8 (Blackwell). Templates: `nvidia/cuda:12.8.0-base-ubuntu22.04`; torch cu128 wheels in `submission-gpu.Dockerfile`; cudnn-runtime tag (+~3 GB) if TensorRT / onnxruntime-gpu are needed | `SUBMISSION.md` §1, `docker/submission.Dockerfile`, `docker/submission-gpu.Dockerfile` |
| Image size | **<= 40 GB uncompressed** | `SUBMISSION.md` §1 |
| Startup | Server listening on port 8000 **within 120 s** of `docker run`; no volumes, no arguments | `SUBMISSION.md` §1 |
| Weights | **Inside the image**, no downloads at startup | `SUBMISSION.md` §1 |
| Network | None provided; external APIs at own risk, a failed call counts as a failure, no reruns. Must have an offline fallback | `FAQ.md`, `RULES_AND_SCORING.md` §7, `SUBMISSION.md` §1 |
| Control rate | 30 Hz loop; sim gate needs **>= 27 Hz of own targets**, in sim clock AND in own wall time (`policy_hz_rt`). A 1-step policy must answer well under 33 ms; slow thinking goes in a background thread while returning hold targets | `RULES_AND_SCORING.md` §4, `INTERFACE.md` §3 |
| Max time per call | 30 s per `act()` call (then `policy_timeout`); 1 s of error replies = watchdog stop | `INTERFACE.md` §3, `SUBMISSION.md` §2 |
| Episode | 120 s, 3-5 objects; finals settings not announced | `RULES_AND_SCORING.md` §2, §5 |
| Pretrained models / data | Any public model or dataset allowed; no training or tuning on the held-out sim variant or its seeds | `FAQ.md`, `RULES_AND_SCORING.md` §7 |
| Sensors | `head_rgb` 480x640 RGB, `head_depth` uint16 mm aligned (0 = invalid), two wrist RGB; no force / tactile | `INTERFACE.md` §2 |

**What this means for perception and language.**
- The perception look runs in the planning thread while the arms are still (baseline `policy.py`, threaded), so it
  is **not** bound by the 33 ms tick. Its budget is episode time: 3-5 looks plus re-looks in 120 s. The CONTEXT.md
  proposal of <= 200 ms per look on CPU is the right order; 15 ms (STATE.md) is what v2 measures, not a limit.
- A GPU model is allowed (RTX 5090, 32 GB). The baseline image is CPU-only at 1.46 GB, so any GPU candidate
  (detector, open-vocab, VLM, local LM) costs image size and startup time: a torch cu128 image is ~6-10 GB, a 7B VLM
  in bf16 ~15 GB of weights. All fit in 40 GB / 32 GB VRAM, but **startup <= 120 s** limits how big a model can be
  loaded (weights must be read from the image and moved to the GPU).
- No network: a hosted VLM or LLM is out as the main path; local weights only.
- The language parser runs once per episode at reset, so even a ~1-2 s local LM call is affordable if it runs
  before the first `act()` returns a target, or in the background thread.

Status: **done** (2026-11-14).

## Vision comparison (A2, A3, A4)

Columns from `eval_vision.py` per family. `unk` = reported `None` (costs 0), `wrong` = a different value (costs -1).

| Approach (`perception.pipeline` / variant) | Commit | Data | recall | FP | colour (unk / wrong) | shape (unk / wrong) | xy med / p95 mm | ms CPU med | Status |
|---|---|---|---|---|---|---|---|---|---|
| `v2` (reference) | daac0e0 | synthetic fresh_06 (120 frames, regression) | 100 % | 0 | 100 % | 99.4 % (3 / 0) | 1.0 / 1.8 | 14.2 cold / 10.1 warm | chosen (synthetic only) |
| `legacy` | daac0e0 | synthetic holdout (240) | 88.3 % | 6.1 % | 99.4 % | 0 % (all unk) | 10.3 / 21.0 | 38.5 | not in use (kept as fallback) |
| `v2` before A2 | 82fec0a | synthetic `a2_close_fresh_01` (60, measured once) | 86.4 % | 0 | 99.5 % (0 / 1) | 83.3 % (35 / 0) | 1.1 / 21.9 | 13.5 | replaced by v2 + A2 |
| `v2` before A2 | 82fec0a | synthetic `a2_occl_fresh_01` (60, measured once) | 100 % | 2 | 100 % (0 / 0) | 92.4 % (13 / **4**) | 1.0 / 8.4 | 14.9 | replaced by v2 + A2 |
| **`v2` + A2** (neck split, wide refit, hidden gate) | 2bdfb87 | synthetic `a2_close_fresh_02` (60, measured once) | 98.8 % | 0 | 100 % (0 / 0) | 95.6 % (11 / **0**) | 1.0 / 2.0 | 32.5 | chosen (synthetic only) |
| **`v2` + A2** | 2bdfb87 | synthetic `a2_occl_fresh_02` (60, measured once) | 100 % | 0 | 100 % (0 / 0) | 92.9 % (16 / **0**) | 1.0 / 2.1 | 22.0 | chosen (synthetic only) |
| `v2` + A2 | 2bdfb87 | synthetic regression_06 (fresh_01..06, 720) | 99.9 % | 3 | 99.9 % (0 / 4) | 98.7 % (36 / 0) | 1.0 / 1.8 (max 4.6) | 20.7 | regression check (before A2: 99.8 %, 1 FP, 15 unk, max 27.4 mm, 14.3 ms) |

### A2. The two open vision failures (done 2026-11-14, synthetic)

Slices from `synth_extra.py` (frozen in b98b5ad before measuring): `close` = two same-colour objects 4.7-6.0 cm
apart; `occl` = one object under a dark occluder, hidden fraction drawn uniformly in 0.3-0.9. Each: dev (tuning),
fresh_01 (scored once on the old code), fresh_02 (scored once on the new code). Per-object A2 report
(`eval_vision.a2_report`):

| Group | Code | n | found | xy p95 / max mm | xy > 10 mm | colour unk / wrong | shape unk / wrong |
|---|---|---|---|---|---|---|---|
| close pair members | before (fresh_01) | 120 | 86 | 24.3 / 28.5 | 32 | 0 / 1 | 37 / 0 |
| close pair members | **after (fresh_02)** | 120 | 118 | 9.1 / 20.0 | 4 | 0 / 0 | 14 / **0** |
| occluded <= 50 % | before (fresh_01) | 25 | 25 | 13.3 / 14.8 | 10 | 0 / 0 | 12 / **4** |
| occluded <= 50 % | **after (fresh_02)** | 15 | 15 | 5.3 / 6.6 | 0 | 0 / 0 | 9 / **0** |
| occluded > 50 % | before (fresh_01) | 35 | 29 | 22.1 / 24.2 | 25 | 0 / 0 | 27 / 0 |
| occluded > 50 % | **after (fresh_02)** | 45 | 39 | 12.2 / 14.7 | 10 | 0 / 0 | 36 / **0** |

Causes and fixes (config `perception.shape`): (1) two same-height top faces < 1 cm apart are bridged by mixed-depth
pixels, so the 3D split saw one object: a top face longer than one object can be (`split_extent` 70 mm; cube
diagonal 63.6 mm) is cut at its least dense slice, and each part is fitted with the other part's cells unknown
(`split_restrict`). (2) a partly hidden object's visible-point centroid is up to ~2 cm off, beyond the +-3 mm fit
search, so a cube with a hidden corner fitted a triangle best: outlines with < 97 % visible are refitted on a
larger grid with +-14 mm search (`polygon.refit_hidden`) and need IoU >= 0.72 and margin >= 0.08 for a shape.
Cost: more abstentions on ordinary frames (shape unk 15 -> 36 of 2809 on regression_06) and +6 ms median.
Open: > 50 % hidden objects still have p95 12 mm position error (shape abstained; a grasp may miss -> 0, the
A5 re-look retries); 2 close-pair members missed in fresh_02.

## Language comparison (A1)

| Parser | Commit | Data | exact | partial | wrong | ms / instruction | Fits A0 | Status |
|---|---|---|---|---|---|---|---|---|
| rule (`instruction.py`) | cd30147 | round_09 + round_10 (fresh when measured) | 56 + 54 / 120 | 4 + 6 | 0 + 0 | ~1 | yes (stdlib) | earlier result |
| **rule** (`instruction.py`, parser unchanged since cd30147) | 64d845d | **a1_fresh_01 + a1_fresh_02** (blind, measured once) | 52 + 51 / 120 (86.7 / 85.0 %) | 4 + 7 | **4 + 2** (non-adv 1 + 1) | 0.7-1.0 | yes (stdlib, CPU) | **chosen** |
| LM: Qwen2.5-3B-Instruct, JSON validated to the 9-type table | 64d845d | same | 27 + 24 (45.0 / 40.0 %) | 11 + 10 | **22 + 26** | ~1,130 (GPU 0, shared) | fits (6 GB bf16, 32 GB GPU; adds torch + weights to the image, < 120 s start) | dropped: unsafe |
| hybrid: rules, LM only for types the rules left unmapped, kept only if it agrees on every resolved type | 64d845d | same | 42 + 42 (70.0 / 70.0 %) | 1 + 3 | **17 + 15** | ~800 | fits (as LM) | dropped: more wrong bins than rules |

### A1. Language (done 2026-11-14)

Two fresh batches (60 each: 30 colour, 12 shape, 8 leave, 10 adversarial) written by a separate agent that was
not allowed to read any batch, the parser or the repo; frozen in 64d845d before any parser ran on them. All three
parsers scored once with `eval_lang.py --parser rule|lm|hybrid`.

**Result:** the rule parser stays the choice. The LM confuses colour and shape scopes and invents bins for types
the text leaves alone (48 wrong in 120). The hybrid as specified fails for a structural reason: "unmapped by the
rules" includes the types an instruction deliberately leaves on the counter, the model fills them, and the
agreement check cannot catch it because the rules have no opinion on those types. A model is not worth adding.

**But the language stop rule does not hold on truly blind data:** the rule parser made 6 wrong bins in 120 (the
generator of rounds 1-10 had been told the fixed categories). All six are real parser errors (labels checked):
quoted rule revoked with "Ignore that; red goes right now, yellow goes where red was supposed to go"
(a1f1_06); "the first bin from the left ... the second ... the third" (a1f1_36); "every cube goes left, unless it is
blue" scoping blue to cubes (a1f1_42); "blue is not to go anywhere" (a1f1_49); "yellow goes nowhere" (a1f2_45);
header "Triangles: blue ones left, ..." scoping to triangles (a1f2_47). These batches are now regression; fixing
them is a new language round with new blind batches.

## Closed loop (A5) (done 2026-11-14, mock world)

`loop/closed_loop.py`: parser (instruction.py) + perception + policy in the mock world over a FIXED list: seeds
500-539, instruction k = fixed interleave of round_09 / round_10 items (true 9-type mapping) and training templates.
The policy sees only the text; scoring on the true mapping with scene.score. Re-look after every place:
`ClassicalPolicy._check_last_place` (still_on_counter / dropped_elsewhere / left_counter); a missed grasp is then
retried by the existing loop (max_attempts). Self-check with known truth: `test_classical.TestReLook` (mock hand
that never grasps -> every check `still_on_counter`, objects on the counter; normal -> `left_counter`, full score).

| Stack | Commit | Data | Score | Objects | Wrong bins | Re-look agrees with world | Status |
|---|---|---|---|---|---|---|---|
| rule parser + v2 (+A2) | (this commit) | mock world, 40 episodes | **296 / 304 (97.4 %)** | 148 correct, 9 left on counter | 0 | 148 / 148 | chosen |
| rule parser + v2-colour-invariant | (this commit) | mock world, same 40 | 296 / 304 | same | 0 | 148 / 148 | same (mock light has no cast) |

All 8 missing points are 2 adversarial instructions the parser leaves partial (r10_60 "both yellow and triangular",
r09_56 "treat the yellow objects as if they were blue"): objects correctly left alone (0), never a wrong bin.
Batch non-adversarial 198/198, templates 78/78. The mock world has kinematic grasping, so the re-look only ever saw
`left_counter` here; the miss path is covered by the self-check.

### A3 so far: CPU candidates on the harder `shift` family (synthetic)

`shift` (synth_extra.py): strong coloured light (deep tungsten, cyan, green, magenta), very dim / over-bright
exposure, mats in the object hues, larger object colour shifts. v2 breaks on it (36 wrong colours on dev), so
candidates can show a gain. Scored ONCE on `shift_fresh_01` (80 frames, commit ebf324c; colour-invariant without
the later cast gate):

| Approach | recall | FP | colour (unk / wrong) | shape (unk / wrong) | xy med / p95 | ms cold | pts/obj type / colour | Note |
|---|---|---|---|---|---|---|---|---|
| `legacy` | 46.2 % | 66 | 86.9 % (0 / 19) | 0 % | 11.7 / 25.2 | 30.2 | -0.06 / +0.53 | not in use |
| `v2` (+A2) | 85.7 % | 1 | 88.8 % (0 / **30**) | 97.0 % (8 / 0) | 1.0 / 1.8 | 19.1 | +1.38 / +1.42 | reference |
| `v2-colour-invariant` | 97.8 % | 0 | 94.8 % (0 / **16**) | 97.1 % (9 / 0) | 1.0 / 1.8 | 25.8 | **+1.75 / +1.80** | best; rescore in B1 |
| `cloud` | 98.1 % | 1 | 77.6 % (39 / 30) | 97.4 % (8 / 0) | 1.0 / 1.8 | 21.9 | +1.38 / +1.42 | finds the uncoloured objects (as obstacles), no points gained |
| `cloud` + colour-invariant | 99.7 % | 0 | 93.0 % (6 / 16) | 97.1 % (9 / 0) | 1.0 / 1.8 | 25.8 | +1.75 / +1.80 | same points, all objects known to the planner |

Regression_06 (standard families, 720 frames) with colour-invariant + cast gate (`wb_min_cast` 1.3, commit
54d18f4): colour wrong 8 vs 4 for v2, pts/obj +1.960 vs +1.967: a small loss on ordinary light for +0.4 per object
under strong casts. `cloud` as implemented is v2 plus keeping uncoloured objects with a clear prism fit: v2 is
already geometry-first (depth segmentation), so a separate point-cloud pipeline would duplicate it.

### A3 GPU candidates (GPU 0, approved by manh 2026-11-14; synthetic)

Implemented behind the same Detection output (`perception.segmenter: detector | open_vocab`,
`perception.vlm_verify`; code `loop/vision/learned.py`): the model gives regions (+ colour / shape labels), v2's
depth part still gives position, yaw and the shape fit; the model's shape is used only where the fit cannot decide.
- `detector`: YOLO11n-seg (public COCO weights, 2.9 M params, 6 MB) fine-tuned 60 epochs (8 min) on 2,400 rendered
  frames (`detector_data.py`, seeds 3,000,000+, families train / unseen / shift, layouts band / wide / close /
  occl, 9 classes colour x shape). Synthetic val mask mAP50 0.99. ~11 ms GPU inference.
- `open-vocab`: OWLv2 base (google/owlv2-base-patch16-ensemble), 9 text queries, boxes -> object-height pixels.
- `vlm-verify`: Qwen2.5-VL-3B-Instruct names colour + shape on a 112 px crop, only for detections with shape None
  or colour vote < 0.9; it may fill a missing shape, a disagreeing colour becomes None.

Scored once:

| Approach | Data | recall | FP | colour (unk / wrong) | shape (unk / wrong) | xy med / p95 | ms med (GPU) | pts/obj type | Note |
|---|---|---|---|---|---|---|---|---|---|
| `detector` | shift_fresh_01 | 100 % | 0 | 98.4 % (0 / **5**) | 100 % (0 / 0) | 1.1 / 2.2 | 37.9 | **+1.95** | best synthetic; IN-DISTRIBUTION (trained on the same renderer, other seeds) |
| `open-vocab` | shift_fresh_01 | 99.0 % | 13 | 68.8 % (0 / **97**) | 95.8 % (13 / 0) | 1.1 / 2.2 | 76 | +1.02 | colour naming poor |
| `vlm-verify` | shift_fresh_01 | 85.7 % | 1 | 88.5 % (2 / 29) | 97.4 % (7 / 0) | 1.0 / 1.8 | 33 (p95 273) | +1.38 | = v2: v2's colour errors are confident, never verified |
| `detector` | regression_06 | 100 % | 2 | 100 % (0 / 0) | 100 % (0 / 1) | 1.0 / 2.1 | 38 | +1.999 | |
| `open-vocab` | regression_06 | 99.8 % | 73 | 92.4 % (0 / 214) | 96.6 % (92 / 4) | 1.0 / 2.0 | 74 | +1.73 | |
| `vlm-verify` | regression_06 | 99.9 % | 3 | 99.8 % (1 / 4) | 99.3 % (18 / **2**) | 1.0 / 1.8 | 30 (p95 154) | +1.98 | VLM adds wrong shapes |

A0 fit: detector 6 MB + torch (~6 GB image), fits; OWLv2 0.6 GB; Qwen2.5-VL-3B ~7.5 GB bf16, fits 32 GB, load
time to be checked against the 120 s start limit.

### A4. When to abstain (done 2026-11-14, synthetic)

`a4_sweep.py` applies thresholds after detection to the stored per-object confidences and maximises expected
points per object (+2 right, -1 wrong, 0 abstained; FP -1). Plot: `results/a4_shift_dev.png`, curves in
`results/a4_shift_dev.json` (hardest family, shift_dev).

| Variant | colour threshold at max | shape extra margin at max | pts/obj type at 0 -> at max | colour wrong at 0 -> at max |
|---|---|---|---|---|
| v2, conf = vote share | 0.55 | 0.00 | +1.401 -> +1.407 | 36 -> 34 |
| v2, conf = share x hue margin | 0.125 | 0.00 | +1.401 -> +1.404 | 36 -> 35 |
| colour-invariant, vote share | 0.00 | 0.00 | +1.780 -> +1.780 | 13 -> 13 |
| colour-invariant, share x hue margin | 0.125 | 0.00 | +1.780 -> +1.783 | 13 -> 12 |

**Finding:** the wrong colours under strong casts are confidently wrong (whole object shifted across a hue
boundary; many have vote share 1.0), so no per-object threshold separates them: the curves are flat and the gain
is <= 0.006 points per object. Chosen: `colour.min_conf: 0.0` (abstain off) and no extra shape margin (shape wrong
is already 0; the fit's own gates abstain). The lever against casts is frame-level (colour-invariant), not
per-object abstention. Repeat on Isaac frames after B1.

Camera pose error beyond the train range (`pose_<k>` slices: train family with uniform +-k cm and +-k deg per axis,
30 band frames each, v2):

| +- cm / deg | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| recall | 100 % | 100 % | 73.9 % | 36.5 % | 9.3 % | 9.0 % |
| FP | 0 | 0 | 74 | 117 | 120 | 58 |
| xy p95 mm | 1.5 | 1.6 | 13.4 | 29.0 | 28.0 | 28.4 |

**Break point: between +-2 and +-3** (cm and deg together). Widening the refinement guards (tilt / yaw 7 deg,
shift 7 cm, more ICP iterations) does not move it (pose_3: recall 76 %, 48 FP): beyond ~2 deg the nominal-camera
object window and the plane candidate band no longer contain the counter / bin rims, so refinement starts from a
wrong guess. The real head pitch must be within ~2 deg of calibration.yaml: the M7 AprilTag extrinsics check is
needed before the real robot (see B3).

## All approaches on the new REAL data (2026-11-14, evidence: `real`)

Data: `/workspace/data/real_drops/frames/g1_junction_drop_2` (added by long on 2026-11-14: start / mid / end head frames of
48 LeRobot teleop episodes). 16 start frames (episodes 0, 3, ..., 45) labelled by eye (`real/drop2_labels.json`:
colour, shape, pixel centre; 61 countable objects, 9 more cut by the image edge). Tool: `eval_real.py` (match by
blob pixel within 40 px). Results: `results/real_drop2.json`.

**How this data differs from what the stack was built for:** no depth stream in these drops (zeros), so every
pipeline runs its depth-free path; the camera pose is not the calibrated one (later episodes higher and further
back); objects are orange / purple / yellow foam blocks, not red / yellow / blue (scored with orange -> red,
purple -> blue); plain wooden counter, no printed mat; robot hands, orange tape and glare in view.

| Approach | recall | FP (16 frames) | colour ok / unk / wrong | shape ok / unk / wrong (known) | pts / obj | ms med | colour errors |
|---|---|---|---|---|---|---|---|
| `legacy` | 31/61 (50.8 %) | 174 | 28 / 0 / 3 | 0 / 31 / 0 (31) | -2.90 | 22 | yellow->blue 3 |
| `v2` (+A2) | 28/61 (45.9 %) | 163 | 24 / 0 / 4 | 0 / 26 / 1 (27) | -2.74 | 31 | yellow->blue 3, orange->blue 1 |
| `v2-colour-invariant` | 28/61 | 163 | 24 / 0 / 4 | 0 / 26 / 1 | -2.74 | 29 | same as v2 |
| `cloud` + colour-invariant | 28/61 | 163 | 24 / 0 / 4 | 0 / 26 / 1 | -2.74 | 29 | same (needs depth) |
| **`detector`** (synthetic-trained YOLO11n-seg) | 39/61 (63.9 %) | **4** | 25 / 0 / **14** | **35** / 0 / 3 (38) | **+0.43** | 18 (GPU) | orange->yellow 14 |
| `open-vocab` (OWLv2) | **46/61 (75.4 %)** | 8 | 30 / 0 / **16** | 32 / 9 / 3 (44) | +0.26 | 60 (GPU) | orange->yellow 16 |
| `open-vocab`, real colour words (orange / purple) | 46/61 | 9 | 28 / 0 / 18 | 22 / 19 / 3 (44) | -0.11 | 57 | orange->yellow 16, purple->yellow 2 |
| `vlm-verify` | 28/61 | 163 | 24 / 1 / 3 | 12 / 12 / 3 (27) | -2.36 | 1,244 | (v2's FPs untouched) |

**What this says.**
1. Without depth, every colour-threshold pipeline is unusable on real frames: ~10 false positives per frame (orange
   bins and tape, glare, hand edges) at -2.7 to -2.9 points per object. The real robot has the D435i depth stream
   (INTERFACE.md), so this is the depth-free fallback failing, not v2's main path; but it means v2 has never been
   tested on real depth, and a dropped depth stream at the venue would be catastrophic for it.
2. The learned segmenters are the only approaches that hold up on real images without depth: 4-8 FP in 16
   frames, shapes mostly right (detector 35 / 38, trained on synthetic frames only).
3. Their failure is colour: every learned colour error is orange read as yellow. With the competition's red objects
   this may not occur, but it shows colour must not come from the model's class label alone.
4. Not the official result: hand labels, 16 frames, wrong colours of objects, no depth, uncalibrated pose.

**Next step suggested by the data (not run):** learned regions (detector or open-vocab) for objecthood + shape,
colour from a per-frame hue clustering of the region pixels against the known object set, depth fit for position
when depth exists. Rescore on Isaac frames (B1) and on a drop WITH depth.

## Decision status (2026-11-14)

- Language: rule parser (A1). Open: 6 new wrong constructions found by blind batches.
- Vision default stays `v2` + A2 for sim (depth present; best measured on synthetic frames with depth together with
  `v2-colour-invariant`); NOT chosen for the finals yet. Candidates to rescore in B1: `v2-colour-invariant`,
  `detector`, `open-vocab`. Real frames argue for a learned segmenter (see above).
- Track B (B1-B3) not run: no Isaac slot (runbook `TRACK_B_RUNBOOK.md`).

## Detector + hue colour hybrid (2026-11-14)

Built from the real-frame finding above (learned regions are right, their colour labels read orange as yellow).
`perception.segmenter: detector` + `perception.colour_from: hue_cluster` (code: `learned.region_hue`,
`learned.assign_colours`, `Perception._hue_colours`; self-check `test_perception.TestHueHybrid`):
1. regions + shape from the YOLO11n-seg detector, position / yaw from the depth fit when depth exists (as before);
2. per region: saturation-weighted circular mean hue of its well-exposed coloured pixels (glare, shadow, grey and
   bin-hull pixels dropped);
3. per frame: region hues clustered (gap 10), clusters assigned to colours by minimum total hue distance with each
   colour used once (Hungarian): the hue ORDER decides, so orange stays "red" next to a yellow object even where
   nearest-prototype would say yellow; > 25 hue units from the assigned prototype -> None.
Options: `hue.prototypes` (`scene` = scene.yaml colours; `real_drop2` = {red 12.4, yellow 41.4, blue 110.0},
median region hue of the labelled objects in the 16 dev frames), `hue.white_balance` (bin white balance first),
`hue.agree_with_model` (keep a colour only if the detector's label agrees, else None).

Honesty note: the design was made after seeing the 16 labelled drop-2 frames, which are therefore dev. The fresh
real test is 8 new drop-2 frames + 4 pick_cube frames labelled and committed (2685552) before any scoring.

| Variant | real dev (16, pts/obj, colour wrong) | **real fresh drop2 (8)** | **real fresh pick_cube (4)** | synth shift_fresh_02 | synth regression_06 |
|---|---|---|---|---|---|
| `v2` | -2.74, 4 | -2.97 (88 FP) | -1.22 (28 FP) | +1.28, 37 | +1.967, 4 |
| `detector` | +0.43, 14 | +0.50, 4 | +0.87, 5 | **+1.95, 3** | **+1.999, 0** |
| `open-vocab` | +0.26, 16 | +0.23, 9 | -0.13, 7 | - | +1.73, 214 |
| **`detector` + hue (scene prototypes)** | +0.93, 0 | +0.70, 2 | **+1.26, 0** | +1.39, 34 | +1.981, 12 |
| `detector` + hue + white balance | +0.90, 0 | +0.57, 2 | +1.26, 0 | (dev +1.75, 14) | - |
| `detector` + hue + wb + agree | +0.51, 0 | +0.47, 1 | +1.00, 0 | (dev +1.75, 0) | - |
| **`detector` + hue, real prototypes** | +0.89, 1 | **+0.80, 1** | **+1.26, 0** | - | - |

Fresh real recall: drop2 16/30 (hands hide many objects in later episodes), pick_cube 22/23; FP 2 and 0 for all
detector variants. Remaining real errors are shapes (pick_cube: 5 cylinders read as cubes) and missed hidden objects.

**Conclusion:** on real frames the hybrid is the best approach measured (fresh: +0.80 and +1.26 points per object,
1 and 0 wrong colours, vs v2 at -2.97 / -1.22 without depth). On synthetic strong colour casts raw hue breaks
(34 wrong) where the in-distribution detector does not; white balance halves that but costs a little on real
frames. Chosen for real-robot work: `detector_hue_real` (prototypes calibrated from a venue photo, as scene.yaml
already asks for colours). Sim default unchanged until B1 rescoring. Caveats as before: no depth in these drops,
hand labels, small sample (12 fresh frames, 53 objects).
