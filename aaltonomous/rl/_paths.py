"""sys.path setup for the RL env: upstream unitree_sim_isaaclab on the path, but WITHOUT running its
tasks/__init__.py (it imports every task package, which starts a DDS participant on domain 1 = the live eval sims).
A bare `tasks` package is registered so `tasks.common_config` (robot + camera presets) imports alone."""
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UP = REPO / "third_party/unitree_sim_isaaclab"
os.environ.setdefault("PROJECT_ROOT", str(UP))      # upstream robots/unitree.py reads the USD paths from it
for p in (str(REPO), str(UP)):
    if p not in sys.path:
        sys.path.insert(0, p)
if "tasks" not in sys.modules:
    _m = types.ModuleType("tasks")
    _m.__path__ = [str(UP / "tasks")]
    sys.modules["tasks"] = _m
