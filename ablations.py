import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def ema(xs, alpha=0.1):
    out = [xs[0]]
    for x in xs[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return np.array(out)


def load_run(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for k, v in row.items():
                data[k].append(v)
    return data


runs = {}
for filename in os.listdir("runs"):
    if filename.endswith(".jsonl"):
        runs[filename[:-6]] = load_run(f"runs/{filename}")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
(
    ax_loss_steps,
    ax_loss_tokens,
    ax_throughput,
    ax_instability,
    ax_max_instability,
    ax_empty,
) = axes.flatten()

# ---------- Loss vs steps ----------
for name, run in runs.items():
    ax_loss_steps.plot(run["step"], ema(run["train_loss"]), label=name)

ax_loss_steps.set_title("Loss vs Steps")
ax_loss_steps.set_xlabel("Steps")
ax_loss_steps.set_ylabel("Train Loss (EMA)")
ax_loss_steps.legend(fontsize=8)

# ---------- Loss vs tokens ----------
for name, run in runs.items():
    tokens = run["total_tokens"]

    ax_loss_tokens.plot(tokens, ema(run["train_loss"]), label=name)

ax_loss_tokens.set_title("Loss vs Tokens Seen")
ax_loss_tokens.set_xlabel("Tokens")
ax_loss_tokens.set_ylabel("Train Loss (EMA)")

# ---------- Throughput ----------
for name, run in runs.items():
    ax_throughput.plot(run["step"], ema(run["tok_per_sec"]), label=name)

ax_throughput.set_title("Throughput")
ax_throughput.set_xlabel("Steps")
ax_throughput.set_ylabel("Tokens / Sec (EMA)")

# ---------- Instability: |Δloss| ----------
for name, run in runs.items():
    loss = np.array(run["train_loss"])
    delta = np.abs(np.diff(loss))
    ax_instability.plot(run["step"][1:], ema(delta), label=name)

ax_instability.set_title("Instability: |Δ Loss|")
ax_instability.set_xlabel("Steps")
ax_instability.set_ylabel("|Δ Loss| (EMA)")

# ---------- Max instability summary ----------
names = []
max_deltas = []
for name, run in runs.items():
    loss = np.array(run["train_loss"])
    max_deltas.append(np.max(np.abs(np.diff(loss))))
    names.append(name)

ax_max_instability.bar(names, max_deltas)
ax_max_instability.set_title("Worst-Case Instability")
ax_max_instability.set_ylabel("Max |Δ Loss|")
ax_max_instability.tick_params(axis="x", rotation=45)

# ---------- Empty / future ----------
ax_empty.axis("off")

description = (
    "What these plots show:\n\n"
    "Loss vs Steps:\n"
    "  How training progresses over optimizer steps.\n"
    "  Useful for debugging schedules.\n\n"
    "Loss vs Tokens:\n"
    "  Main comparison plot.\n"
    "  Lower loss at the same tokens = better model.\n\n"
    "Throughput:\n"
    "  How fast each model trains.\n\n"
    "|Δ Loss| (Instability):\n"
    "  Measures how jumpy training is.\n"
    "  Big spikes mean instability.\n\n"
    "Max |Δ Loss|:\n"
    "  Single worst jump during training."
)

ax_empty.text(
    0.0,
    1.0,
    description,
    va="top",
    ha="left",
    fontsize=11,
)


plt.tight_layout()
plt.show()
