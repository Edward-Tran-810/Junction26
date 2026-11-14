"""PERCEPTION_MILESTONES A1: a small LOCAL language model as instruction parser, and a rules + model hybrid.

    parse_lm(text)      -> mapping {(colour, shape): bin} over the 9 types (unmapped types absent)
    parse_hybrid(text)  -> rule parse; the model is consulted only for types the rules left unmapped, and its answer
                           is kept only if it agrees with the rules on EVERY type the rules resolved

Model: Qwen2.5-3B-Instruct (public weights, bf16, ~6 GB; runs on one GPU, fits the 32 GB finals GPU, see A0), greedy
decoding. Output is constrained by validation, not trust: the model must return one JSON object whose keys are
exactly the 9 "colour/shape" types and whose values are "left" | "center" | "right" | null. Anything else (bad JSON,
extra / missing keys, other values) -> empty mapping (everything stays on the counter: 0 points, never -1).

Env: AALTO_LM_MODEL (default Qwen/Qwen2.5-3B-Instruct), AALTO_LM_DEVICE (default cuda:0).
"""
from __future__ import annotations

import json
import os
import re
import time

from aaltonomous.baselines.classical.instruction import COLORS, SHAPES, expand, parse_instruction

TYPES = [f"{c}/{s}" for c in COLORS for s in SHAPES]
BINS = ("left", "center", "right")

SYSTEM = """You convert a robot sorting instruction into a table. Objects: colours red, yellow, blue; shapes cube, \
cylinder, triangle (a triangular prism, "tent", "wedge"). Bins are named only by position: left, center (also \
"middle"), right. "box" means bin.
Return ONE JSON object with exactly these 9 keys: """ + ", ".join(f'"{t}"' for t in TYPES) + """.
Each value is "left", "center", "right", or null. null means the instruction does not send that object type to a \
bin (it stays on the table), or you cannot tell. Never guess: if unsure, use null. Later corrections override \
earlier statements. Output only the JSON, no explanation."""

FEWSHOT = [
    ("Blue things go left; put every cylinder in the middle no matter its colour. Ignore the rest.",
     {**{t: None for t in TYPES}, "blue/cube": "left", "blue/triangle": "left",
      "red/cylinder": "center", "yellow/cylinder": "center", "blue/cylinder": "center"}),
    ("yellow -> right bin, red -> left bin, and the blue ones in the centre one",
     {t: {"red": "left", "yellow": "right", "blue": "center"}[t.split("/")[0]] for t in TYPES}),
]

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = os.environ.get("AALTO_LM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        dev = os.environ.get("AALTO_LM_DEVICE", "cuda:0")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
        _MODEL = (tok, model, dev)
    return _MODEL


def raw_lm(text: str, max_new_tokens: int = 200) -> str:
    import torch
    tok, model, dev = _load()
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
    msgs.append({"role": "user", "content": text})
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt").to(dev)
    n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True)


def validate(raw: str):
    """Strict check of the model output -> {(c, s): bin} or None (invalid)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or set(d) != set(TYPES):
        return None
    syn = {"middle": "center", "centre": "center"}           # exact bin-name synonyms only
    d = {k: syn.get(v, v) if isinstance(v, str) else v for k, v in d.items()}
    if any(v not in BINS and v is not None for v in d.values()):
        return None
    return {tuple(k.split("/")): v for k, v in d.items() if v is not None}


LAST = {}


def parse_lm(text: str) -> dict:
    t0 = time.perf_counter()
    raw = raw_lm(text)
    m = validate(raw)
    LAST.update(raw=raw, valid=m is not None, ms=(time.perf_counter() - t0) * 1000)
    return m or {}


def parse_hybrid(text: str) -> dict:
    rules = expand(parse_instruction(text))                     # 9 types, None = unmapped
    if all(v is not None for v in rules.values()):
        LAST.update(raw=None, valid=True, ms=0.0, consulted=False)
        return {k: v for k, v in rules.items()}
    lm = parse_lm(text)
    LAST["consulted"] = True
    if not LAST["valid"]:
        return {k: v for k, v in rules.items() if v is not None}
    # keep the model only if it agrees with the rules on every type the rules resolved
    if any(lm.get(k) != v for k, v in rules.items() if v is not None):
        LAST["agreed"] = False
        return {k: v for k, v in rules.items() if v is not None}
    LAST["agreed"] = True
    out = {k: v for k, v in rules.items() if v is not None}
    out.update({k: v for k, v in lm.items() if rules[k] is None})
    return out
