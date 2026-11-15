"""Dex3 contact stream client (sim: PhysX contact sensors, sim/aalto_task/hooks.py `_publish_contacts`).

The sim publishes one JSON message per physics env step (100 Hz) on zmq PUB :$AALTO_CONTACT_PORT (default scene port + 2)
when it was started with AALTO_HAND_CONTACTS=1:
    {"t": sim_t, "c": {"left_palm": {"net": N, "obj": {"obj_red_cube_0": N}, "other": N, "fz_obj": N}, "left_thumb_2": ...}}
Forces are PhysX NORMAL contact forces (newtons). `obj` = force against each object, `other` = everything else (table, lid,
bins, the other hand). Links with no contact are omitted. Used by the top-down controller as a tactile stand-in: the
real Dex3 has palm / finger pressure sensors; this is the sim's equivalent and it is NEVER fabricated: no stream ->
`available` is False and the controller falls back to the near-contact height trigger + finger-load verification.

Groups per hand: palm, thumb (thumb_1, thumb_2), index (index_0, index_1), middle (middle_0, middle_1).
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time

GROUPS = {"palm": ("palm",), "thumb": ("thumb_1", "thumb_2"), "index": ("index_0", "index_1"),
          "middle": ("middle_0", "middle_1")}


def contact_port():
    p = os.environ.get("AALTO_CONTACT_PORT")
    if p:
        return int(p)
    if os.environ.get("AALTO_HAND_CONTACTS") == "1":
        return int(os.environ.get("AALTO_SCENE_PORT", "55580")) + 2
    return 0


class ContactStream:
    """Background SUB socket keeping the last `history` messages. Thread-safe reads."""

    def __init__(self, host="localhost", port=None, history=200):
        self.port = contact_port() if port is None else int(port)
        self.buf = collections.deque(maxlen=history)
        self.lock = threading.Lock()
        self.n = 0
        self._stop = False
        self.thread = None
        if self.port:
            import zmq
            self.zmq = zmq
            self.sock = zmq.Context.instance().socket(zmq.SUB)
            self.sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.sock.setsockopt(zmq.RCVHWM, 100)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(f"tcp://{host}:{self.port}")
            self.thread = threading.Thread(target=self._run, daemon=True, name="contact-stream")
            self.thread.start()

    def _run(self):
        poller = self.zmq.Poller(); poller.register(self.sock, self.zmq.POLLIN)
        while not self._stop:
            if not poller.poll(100):
                continue
            while True:
                try:
                    m = json.loads(self.sock.recv(self.zmq.NOBLOCK))
                except self.zmq.Again:
                    break
                except ValueError:
                    continue
                m["wall"] = time.monotonic()
                with self.lock:
                    self.buf.append(m)
                    self.n += 1

    @property
    def available(self):
        """True when messages arrived within the last second."""
        with self.lock:
            return bool(self.buf) and time.monotonic() - self.buf[-1]["wall"] < 1.0

    def recent(self, k):
        with self.lock:
            return list(self.buf)[-k:]

    def close(self):
        self._stop = True


def group_forces(msg, side, objects=None):
    """{group: {"obj": max object force, "objs": {name: N}, "other": N, "net": N}} for one hand in one message.
    objects: restrict the object forces to these names (None = any object)."""
    out = {}
    c = msg.get("c") or {}
    for g, links in GROUPS.items():
        o, other, net, objs, lo, lt = 0.0, 0.0, 0.0, {}, None, None
        for l in links:
            d = c.get(f"{side}_{l}")
            if not d:
                continue
            for name, f in (d.get("obj") or {}).items():
                if objects is None or name in objects:
                    if f > objs.get(name, 0.0):
                        objs[name] = f
                    if f >= o:
                        o, lo = f, f"{side}_hand_{l}_link"
            if float(d.get("other", 0.0)) >= other:
                other, lt = float(d.get("other", 0.0)), f"{side}_hand_{l}_link"
            net = max(net, float(d.get("net", 0.0)))
        out[g] = {"obj": o, "objs": objs, "other": other, "net": net, "obj_link": lo, "other_link": lt}
    return out


def confirmed(msgs, side, group, thr, key="obj"):
    """Group `group` had `key` force >= thr in EVERY one of msgs (noise rejection: k consecutive 100 Hz samples)."""
    return bool(msgs) and all(group_forces(m, side)[group][key] >= thr for m in msgs)
