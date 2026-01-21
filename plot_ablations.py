import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def load_run(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for k, v in row.items():
                data[k].append(v)
    return data


# read all files available in the runs directory
runs = {}
for filename in os.listdir("runs"):
    if filename.endswith(".jsonl"):
        run_name = filename[:-6]  # remove .jsonl
        runs[run_name] = load_run(f"runs/{filename}")

plt.figure()
for name, run in runs.items():
    plt.plot(run["step"], run["train_loss"], label=name)

plt.xlabel("Training Steps")
plt.ylabel("Train Loss")
plt.legend()
plt.title("Training Loss Ablation")
plt.show()


plt.figure()
for name, run in runs.items():
    plt.plot(run["step"], run["tok_per_sec"], label=name)

plt.xlabel("Training Steps")
plt.ylabel("Tokens / Second")
plt.legend()
plt.title("Throughput Ablation")
plt.show()


plt.figure()
for name, run in runs.items():
    plt.plot(run["step"], run["grad_norm"], label=name)

plt.xlabel("Training Steps")
plt.ylabel("Gradient Norm")
plt.legend()
plt.title("Gradient Stability")
plt.show()
