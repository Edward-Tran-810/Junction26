"""Summarise one physical top-down test directory (runs/topdown_contact/<name>): per-pick controller record
(td_picks.jsonl, written by the policy with AALTO_TD_LOG) + the evaluator's scored outcome (eval.json).

Slip = change of the object's position RELATIVE to the hand's grasp point (scene-service ground truth vs FK of the
measured arm joints, AALTO_TD_DIAG=1) from GRASP VERIFIED to the release; rotation = change of the object's orientation
over the same window. Measurement only.
Run: python -m aaltonomous.experiments.topdown_contact.analyse_test runs/topdown_contact/t2_cube"""
import json, sys
from pathlib import Path

import numpy as np


def quat_angle_deg(q1, q2):
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def pick_summary(r):
    D = r.get("diag") or []
    ver = next((d for d in D if d["label"] == "verified"), None)
    after = [d for d in D if ver and d["t"] >= ver["t"]]
    slip = rot = None
    if ver and after:
        slip = max(float(np.linalg.norm(np.subtract(d["rel_mm"], ver["rel_mm"]))) for d in after)
        rot = max(quat_angle_deg(d["quat_xyzw"], ver["quat_xyzw"]) for d in after)
    trig = next((d for d in D if d["label"] == "trigger"), None)
    rise = None
    if trig and after:
        rise = round((after[-1]["obj_pos"][2] - trig["obj_pos"][2]) * 1e3, 1)   # object lifted with the hand (mm)
    return {"shape": r.get("shape"), "color": r.get("color"), "side": r.get("side"), "outcome": r.get("outcome"),
            "trigger": r.get("trigger"), "palm_contact": r.get("palm_contact"), "contact_link": r.get("contact_link"),
            "contact_force_n": r.get("contact_force_n"), "close_s": r.get("close_s"), "groups_at_close": r.get("groups_at_close"),
            "forces_at_closed_n": r.get("forces_at_closed_n"), "hold_torque_nm_max": max(r["hold_torque_nm"]) if r.get("hold_torque_nm") else None,
            "verified_contacts": r.get("verified_contacts"), "regrips": r.get("regrips"), "release": r.get("release"),
            "slip_mm_max": None if slip is None else round(slip, 1), "rotation_deg_max": None if rot is None else round(rot, 1),
            "object_rise_mm": rise, "n_diag": len(D),
            "clearance_model_trigger": {k: v for k, v in (r.get("clearance_model_trigger") or {}).items() if k != "per_link_mm"},
            "heights_at_collision": r.get("collision_heights"), "proprio": r.get("proprio"),
            "contact_events": [(e["t"], e["state"], e["link"], e["partner"], e["force_n"]) for e in (r.get("contact_events") or [])],
            "table_contacts": [e for e in (r.get("contact_events") or []) if e["partner"] != "object"],
            "hand_offset_at_descent": r.get("hand_offset_at_descent")}


def main(d):
    d = Path(d)
    picks = [json.loads(l) for l in (d / "td_picks.jsonl").read_text().splitlines()] if (d / "td_picks.jsonl").exists() else []
    ev = sorted(d.glob("*/eval.json"))
    trials = json.loads(ev[-1].read_text())["trials"] if ev else []
    out = {"picks": [pick_summary(r) for r in picks],
           "trials": [{"seed": t["seed"], "n_correct": t["n_correct"], "n_objects": t["n_objects"], "outcomes": [o["outcome"] for o in t["outcomes"]],
                       "end_reason": t.get("end_reason"), "clips": (t.get("safety") or {}).get("clipped_steps"),
                       "command_hz": t.get("command_hz"), "gate_ok": t.get("gate_ok"), "gate_why": t.get("gate_why"),
                       "sim_s": t.get("wall_s"), "video": str(next(iter(sorted((ev[-1].parent).glob("*.mp4"))), ""))} for t in trials]}
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    for p in out["picks"]:
        print("PICK", json.dumps(p))
    for t in out["trials"]:
        print("TRIAL", json.dumps(t))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
