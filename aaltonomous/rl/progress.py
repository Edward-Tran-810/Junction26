"""Training progress from the tensorboard event files: python -m aaltonomous.rl.progress [runs/rl/<pattern>]"""
import glob
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]


def last(run_dir, keys=("Episode/success", "Episode/lifted_any", "Train/mean_reward", "Perf/total_fps",
                        "Policy/mean_noise_std", "Episode/counter_ticks")):
    ev = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ev.Reload()
    tags = ev.Tags()["scalars"]
    out = {}
    for k in keys:
        m = [t for t in tags if t.endswith(k)]
        if m:
            s = ev.Scalars(m[0])
            out[k.split("/")[-1]] = (s[-1].step, round(s[-1].value, 4))
    return out


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "runs/rl/*")
    for d in sorted(glob.glob(pat)):
        if Path(d).is_dir():
            try:
                print(Path(d).name, last(d))
            except Exception as e:  # noqa: BLE001
                print(Path(d).name, "?", e)
