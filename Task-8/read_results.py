"""Read F1 scores from TensorBoard event files."""

import os
import sys
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Install tensorboard first: pip install tensorboard")
    sys.exit(1)

def read_events(logdir):
    results = defaultdict(list)
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if not f.startswith("events.out"):
                continue
            path = os.path.join(root, f)
            ea = EventAccumulator(path)
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                for event in ea.Scalars(tag):
                    results[tag].append((event.step, event.value))
    return results

if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/"
    results = read_events(logdir)

    if not results:
        print(f"No events found in {logdir}")
        sys.exit(1)

    for tag in sorted(results):
        print(f"\n=== {tag} ===")
        for step, value in sorted(results[tag]):
            print(f"  epoch {step}: {value:.4f}")
