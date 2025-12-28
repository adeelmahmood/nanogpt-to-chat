import json
import os

class MetricLogger:
    def __init__(self, out_dir, exp_name):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{exp_name}.jsonl")

    def log(self, step, **metrics):
        row = {"step": step, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
