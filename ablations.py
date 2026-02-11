import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import fnmatch


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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ablation study plots")
    parser.add_argument(
        "--filter-pattern",
        type=str,
        help='Pattern to filter run names (e.g., "*positioning*")',
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory for plots"
    )
    parser.add_argument(
        "--title", type=str, default="Ablation Study", help="Plot title"
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs", help="Directory containing run logs"
    )
    return parser.parse_args()


def load_runs(runs_dir="runs", filter_pattern=None):
    runs = {}

    # Look for both old format (runs/*.jsonl) and new format (runs/*/train.jsonl)
    if os.path.exists(runs_dir):
        for item in os.listdir(runs_dir):
            item_path = os.path.join(runs_dir, item)

            if item.endswith(".jsonl"):
                # Old format: runs/run_name.jsonl
                run_name = item[:-6]
                log_file = item_path
            elif os.path.isdir(item_path):
                # New format: runs/run_name/train.jsonl (or train.jsonl.jsonl if double extension bug)
                run_name = item
                log_file = os.path.join(item_path, "train.jsonl")
                if not os.path.exists(log_file):
                    # Check for double extension bug
                    log_file = os.path.join(item_path, "train.jsonl.jsonl")
                    if not os.path.exists(log_file):
                        continue
            else:
                continue

            # Apply filter if specified
            if filter_pattern and not fnmatch.fnmatch(run_name, filter_pattern):
                continue

            runs[run_name] = load_run(log_file)

    return runs


def generate_plots(runs, title="Ablation Study", output_dir="."):
    if not runs:
        print("No runs found matching the filter pattern!")
        return

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

    ax_loss_steps.set_title(f"{title} - Loss vs Steps")
    ax_loss_steps.set_xlabel("Steps")
    ax_loss_steps.set_ylabel("Train Loss (EMA)")
    ax_loss_steps.legend(fontsize=8)

    # ---------- Loss vs tokens ----------
    for name, run in runs.items():
        tokens = run["total_tokens"]
        ax_loss_tokens.plot(tokens, ema(run["train_loss"]), label=name)

    ax_loss_tokens.set_title(f"{title} - Loss vs Tokens Seen")
    ax_loss_tokens.set_xlabel("Tokens")
    ax_loss_tokens.set_ylabel("Train Loss (EMA)")

    # ---------- Throughput ----------
    for name, run in runs.items():
        ax_throughput.plot(run["step"], ema(run["tok_per_sec"]), label=name)

    ax_throughput.set_title(f"{title} - Throughput")
    ax_throughput.set_xlabel("Steps")
    ax_throughput.set_ylabel("Tokens / Sec (EMA)")

    # ---------- Grad Norm ----------
    for name, run in runs.items():
        if "grad_norm" in run:
            ax_instability.plot(run["step"], ema(run["grad_norm"]), label=name)

    ax_instability.set_title(f"{title} - Grad Norm")
    ax_instability.set_xlabel("Steps")
    ax_instability.set_ylabel("Grad Norm (EMA)")

    # ---------- Update Ratio (Matrix) ----------
    for name, run in runs.items():
        if "update_ratio_matrix" in run:
            ax_max_instability.plot(
                run["step"],
                ema(run["update_ratio_matrix"]),
                label=name,
            )

    ax_max_instability.set_title(f"{title} - Matrix Update Ratio")
    ax_max_instability.set_xlabel("Steps")
    ax_max_instability.set_ylabel("||ΔW|| / ||W|| (EMA)")

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
        "Grad Norm (Instability):\n"
        "  Measures how jumpy training is.\n"
        "  Big spikes mean instability.\n\n"
        "Update Ratio (Matrix):\n"
        "  Single worst jump during training.\n\n"
        f"Comparing: {list(runs.keys())}"
    )

    ax_empty.text(
        0.0,
        1.0,
        description,
        va="top",
        ha="left",
        fontsize=10,
    )

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to: {output_path}")

    plt.show()


def main():
    args = parse_args()

    print(f"Loading runs from: {args.runs_dir}")
    if args.filter_pattern:
        print(f"Filtering with pattern: {args.filter_pattern}")

    runs = load_runs(args.runs_dir, args.filter_pattern)

    print(f"Found {len(runs)} runs: {list(runs.keys())}")

    if runs:
        generate_plots(runs, args.title, args.output_dir)
    else:
        print("No runs found!")


if __name__ == "__main__":
    main()
