"""PASS / FAIL for one demo run from the simulator's ground truth + an annotated MP4 and screenshots.

    python -m aaltonomous.demos.report <run dir> <policy_server.log> --title T --subtitle S [--extra TEXT] --out DIR

Verdict (all from sim/evaluate.py output, i.e. the simulator's final scene, not from the policy's own beliefs):
  PASS     every object scored as intended (sort: 'correct'; insert: 'inserted'), 0 safety clips, evaluator gate ok, episode 'done'
  FAIL     the run finished but at least one object is wrong / not placed, or clips / gate failed
  BLOCKED  no episode result (sim not reachable, crash, interrupted)
Video: head-camera MP4 as recorded by the evaluator (one frame per control tick, sim time) + text overlay; screenshots are
frames of that camera at the start, after each pick (policy log times) and at the end. NOT the Kit viewport.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("server_log")
    ap.add_argument("--title", default=""); ap.add_argument("--subtitle", default=""); ap.add_argument("--extra", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    ej = run / "eval.json"
    res = {"verdict": "BLOCKED", "why": "no eval.json"}
    if ej.exists():
        j = json.loads(ej.read_text()); t = j["trials"][0]
        oc = t.get("outcomes", [])
        good = "inserted" if t.get("task") == "insert_shapes" else "correct"
        ok = [o["outcome"] == good for o in oc]
        clips = (t.get("safety") or {}).get("clipped_steps")
        if t.get("end_reason") in ("interrupted", "crash", "aborted", "watchdog", "policy_connection_lost", "policy_timeout"):
            res = {"verdict": "BLOCKED", "why": f"episode ended: {t.get('end_reason')}"}
        elif oc and all(ok) and clips == 0 and t.get("gate_ok"):
            res = {"verdict": "PASS", "why": f"{sum(ok)}/{len(ok)} objects {good}, 0 safety clips, gate ok"}
        else:
            res = {"verdict": "FAIL", "why": f"{sum(ok)}/{len(ok)} objects {good}; clips {clips}; gate {t.get('gate_ok')} {t.get('gate_why')}"}
        res.update(outcomes=[(o["name"], o["outcome"]) for o in oc], sim_s=t.get("sim_s"), seed=t.get("seed"),
                   instruction=t.get("instruction"), clips=clips, end_reason=t.get("end_reason"), layout=t.get("layout"),
                   final=[(o["name"], [round(v, 3) for v in o["pos"]]) for o in t.get("final_objects", [])])
    # picks from the policy's own log (proxy, labelled as such)
    picks = []
    sl = Path(a.server_log)
    if sl.exists():
        for line in sl.read_text().splitlines():
            m = re.search(r"\[(classical|verify)[^\]]*\] (.*)", line)
            if m and ("execute" in m.group(2) or "verify" in line or "re-look" in m.group(2) or "ABORT" in m.group(2)):
                picks.append(m.group(2)[:150])
    res["policy_log"] = picks
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\n=== {a.title}: {res['verdict']}  ({res['why']})")
    mp4 = next(iter(sorted(run.glob("*.mp4"))), None)
    if mp4 is None:
        print("no mp4 recorded"); return
    def esc(s):
        return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "").replace("%", "%%")
    import textwrap
    raw = [a.title, a.subtitle] + ([a.extra] if a.extra else []) + [f"RESULT: {res['verdict']} - {res['why']}"]
    lines = [w for r in raw for w in textwrap.wrap(r, 62)]
    vf = ",".join(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=15:fontcolor=white:"
                  f"box=1:boxcolor=black@0.6:boxborderw=4:x=8:y={6 + 21 * i}:text='{esc(s)}'" for i, s in enumerate(lines))
    dst = out / f"{Path(a.out).name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], check=False)
    n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                            "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip() or 0)
    shots = out / "screenshots"; shots.mkdir(exist_ok=True)
    for k, frac in enumerate((0.02, 0.2, 0.4, 0.6, 0.8, 0.99)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-vf", f"select=eq(n\\,{int(frac * max(n - 1, 0))})", "-frames:v", "1",
                        str(shots / f"frame_{k}_{int(frac * 100):02d}pct.png")], check=False)
    print("video:", dst, f"({n} frames)", "| screenshots:", shots)


if __name__ == "__main__":
    main()
