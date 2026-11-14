"""Per-pick report of a sim.evaluate run from its eval.json + the policy server's log.

    python -m aaltonomous.baselines.classical.grasp_report runs/eval/<run> runs/eval_logs/server_<team>.log [...]

What each number is:
  OBSERVED (simulator ground truth, sim/evaluate.py): objects correct / counter / wrong / off, final positions,
      sim time, safety clips + events, end reason, gate.
  POLICY VERDICT (proxy, finger load, grasp_monitor.py): grasp confirmed / failed / unconfirmed, lost in transit,
      release clean / recovered / failed.
  CAMERA RE-LOOK (proxy, perception after the arm is home): left_counter / still_on_counter / dropped_elsewhere.
  DERIVED: false-positive grasp detection = the policy called a pick complete (confirmed + carried + released) but
      the camera found the object still at the grasp spot; completed-but-not-binned = picks called complete minus
      objects the simulator found in their bin (false positives + objects that left the hand but missed the bin).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_server(path):
    """{seed: [pick dict, ...]} in execution order."""
    eps, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.search(r"reset task=\S+ seed=(\d+)", line)
        if m:
            cur = eps[int(m.group(1))] = []          # a re-run of a seed replaces it
            continue
        if cur is None:
            continue
        m = re.search(r"execute (\w+)\s+(\w+)\s+arm -> (\w+) bin", line)
        if m:
            cur.append({"color": m.group(1), "side": m.group(2), "bin": m.group(3), "grasp": None, "transit": None,
                        "release": None, "recoveries": 0, "relook": None})
            continue
        if not cur:
            continue
        p = cur[-1]
        if "grasp confirmed" in line:
            p["grasp"] = "confirmed"
        m = re.search(r"ABORT (\w+)", line)
        if m:
            p["grasp" if m.group(1).startswith("grasp") else "transit"] = m.group(1)
        m = re.search(r"release (clean|recovered|failed) .*recoveries (\d+)", line)
        if m:
            p["release"], p["recoveries"] = m.group(1), int(m.group(2))
            p["transit"] = p["transit"] or "ok"
        m = re.search(r"re-look after place: (\w+)", line)
        if m:
            p["relook"] = m.group(1)
    return eps


def report(run_dir, server_log):
    j = json.loads((Path(run_dir) / "eval.json").read_text())
    picks = parse_server(server_log)
    rows, tot = [], {}

    def add(k, v):
        tot[k] = tot.get(k, 0) + v

    for t in j["trials"]:
        P = picks.get(t["seed"], [])
        oc = [o["outcome"] for o in t.get("outcomes", [])]
        fin = {o["name"]: o["pos"] for o in t.get("final_objects", [])}
        verified = any(p["grasp"] or p["transit"] or p["release"] for p in P)
        conf = sum(p["grasp"] == "confirmed" for p in P)
        complete = sum(p["release"] in ("clean", "recovered") for p in P) if verified else None
        r = {"seed": t["seed"], "objects": len(oc), "correct": oc.count("correct"), "counter": oc.count("counter"),
             "wrong": oc.count("wrong"), "off": oc.count("off"),
             "in_hand_at_end": sum(1 for n, p in fin.items() if p[2] > 0.15),
             "picks": len(P), "grasp_confirmed": conf if verified else None,
             "grasp_failed": sum(p["grasp"] in ("grasp_failed", "grasp_unconfirmed") for p in P) if verified else None,
             "lost_in_transit": sum(p["transit"] == "lost_in_transit" for p in P) if verified else None,
             "release_recovered": sum(p["release"] == "recovered" for p in P) if verified else None,
             "release_failed": sum(p["release"] == "failed" for p in P) if verified else None,
             "false_pos": sum(p["release"] in ("clean", "recovered") and p["relook"] == "still_on_counter" for p in P)
             if verified else None,
             "complete_not_binned": (complete - oc.count("correct")) if verified else None,
             "relook_still": sum(p["relook"] == "still_on_counter" for p in P),
             "relook_dropped": sum(p["relook"] == "dropped_elsewhere" for p in P),
             "sim_s": t.get("sim_s"), "end": t.get("end_reason"),
             "clips": (t.get("safety") or {}).get("clipped_steps"), "safety_events": len(t.get("safety_events", [])),
             "hz": t.get("command_hz"), "rt_hz": t.get("policy_hz_rt"), "gate": "ok" if t.get("gate_ok") else "FAIL",
             "detail": [f"{p['color']}/{p['side']}:{p['grasp'] or '-'}/{p['transit'] or '-'}/{p['release'] or '-'}"
                        f"{'+' + str(p['recoveries']) if p['recoveries'] else ''}>{p['relook'] or '-'}" for p in P],
             "left": [f"{o['name'][4:]}={o['outcome']}@({fin[o['name']][0]:+.2f},{fin[o['name']][1]:+.2f},{fin[o['name']][2]:+.2f})"
                      for o in t.get("outcomes", []) if o["outcome"] != "correct" and o["name"] in fin]}
        rows.append(r)
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("seed", "hz", "rt_hz") and v is not None:
                add(k, v)
    return {"run": Path(run_dir).name, "rows": rows, "total": tot, "summary": j["summary"]}


def show(rep):
    print(f"\n===== {rep['run']}")
    cols = ["seed", "objects", "correct", "counter", "in_hand_at_end", "picks", "grasp_confirmed", "grasp_failed",
            "lost_in_transit", "release_recovered", "release_failed", "false_pos", "complete_not_binned", "sim_s",
            "end", "clips", "safety_events", "rt_hz", "gate"]
    print(" | ".join(cols))
    for r in rep["rows"]:
        print(" | ".join("-" if r[c] is None else str(r[c]) for c in cols))
    t = rep["total"]
    print("TOTAL: " + ", ".join(f"{k}={round(v, 1)}" for k, v in t.items()))
    s = rep["summary"]
    print(f"evaluator: objects correct {s['objects_correct']}/{s['objects']}, all-correct episodes {s['episodes_all_correct']}"
          f"/{s['trials']}, end reasons {s['end_reasons']}, clips {s['safety_clipped_steps_total']}, gate "
          f"{'PASS' if s['gate']['passed'] else 'FAIL ' + str(s['gate']['failed_trials'])}")
    for r in rep["rows"]:
        print(f"  seed {r['seed']}: " + "; ".join(r["detail"]))
        if r["left"]:
            print(f"     not sorted: {', '.join(r['left'])}")


if __name__ == "__main__":
    a = sys.argv[1:]
    for run_dir, log in zip(a[0::2], a[1::2]):
        show(report(run_dir, log))
