import json
import os


class MetricLogger:
    def __init__(self, out_dir, file_name="main"):
        self.is_logging = out_dir is not None
        if not self.is_logging:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, file_name)

    def log(self, step, **metrics):
        if not self.is_logging:
            return
        row = {"step": step, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
