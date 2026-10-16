"""
Aggregate the client-size skew sweep: mean +- std over seeds of accuracy and
per-client privacy, random selection vs CSS, for each partition.

Expects results/sweep/<partition>_s<seed>/summary_<selection>_<mode>.json as
written by main.py.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARTITIONS = [("speaker", "Equal\n(per speaker)"), ("b1.0", "β=1.0"),
              ("b0.5", "β=0.5"), ("b0.3", "β=0.3\n(most skewed)")]
SERIES = {"random": ("Random selection", "#2a78d6"), "css": ("CSS", "#eb6834")}
MODES = ["sigma1.0", "sigma3.0", "nondp"]


def load(root):
    runs = defaultdict(list)   # (partition, selection, mode) -> [summary, ...]
    for path in glob.glob(os.path.join(root, "*_s*", "summary_*.json")):
        part = re.match(r"(.+)_s\d+$", os.path.basename(os.path.dirname(path))).group(1)
        sel, mode = re.match(r"summary_(random|css)_(.+)\.json", os.path.basename(path)).groups()
        with open(path) as f:
            runs[(part, sel, mode)].append(json.load(f))
    return runs


def stat(vals):
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def table(runs):
    lines = ["| Partition | Mode | Selection | Accuracy (last 3 evals) | Worst-case ε | Median ε | corr(ε, size) | seeds |",
             "|---|---|---|---|---|---|---|---|"]
    for part, label in PARTITIONS:
        for mode in MODES:
            for sel in SERIES:
                rs = runs.get((part, sel, mode))
                if not rs:
                    continue
                a = stat([100 * r["acc_last3"] for r in rs])
                row = f"| {label.splitlines()[0]} | {mode} | {sel} | {a[0]:.1f} ± {a[1]:.1f}% |"
                if "privacy" in rs[0]:
                    e = stat([r["privacy"]["eps_max"] for r in rs])
                    m = stat([r["privacy"]["eps_median"] for r in rs])
                    c = stat([r["privacy"]["eps_size_corr"] for r in rs])
                    row += f" {e[0]:.1f} ± {e[1]:.1f} | {m[0]:.1f} ± {m[1]:.1f} | {c[0]:+.2f} |"
                else:
                    row += " – | – | – |"
                lines.append(row + f" {a[2]} |")
    return "\n".join(lines)


def figure(runs, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    x = np.arange(len(PARTITIONS))
    for row, mode in enumerate(["sigma1.0", "sigma3.0"]):
        for col, metric in enumerate(["eps_max", "acc"]):
            ax = axes[row, col]
            for k, (sel, (label, color)) in enumerate(SERIES.items()):
                means, stds = [], []
                for part, _ in PARTITIONS:
                    rs = runs.get((part, sel, mode), [])
                    vals = ([r["privacy"]["eps_max"] for r in rs] if metric == "eps_max"
                            else [100 * r["acc_last3"] for r in rs])
                    means.append(np.mean(vals) if vals else np.nan)
                    stds.append(np.std(vals) if vals else np.nan)
                ax.errorbar(x + (k - 0.5) * 0.12, means, yerr=stds, color=color, linewidth=2,
                            marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                            capsize=3, label=label)
            sigma = mode.replace("sigma", "σ=")
            ax.set_title(f"{'Worst-case client ε' if metric == 'eps_max' else 'Accuracy (%)'}, {sigma}",
                         loc="left", fontsize=11)
            ax.set_xticks(x, [lbl for _, lbl in PARTITIONS])
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#9b9a95")
            ax.tick_params(colors="#52514e")
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Client-size skew: random selection vs CSS (mean ± std over seeds; x = more unequal →)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/sweep")
    args = parser.parse_args()
    runs = load(args.root)
    md = table(runs)
    print(md)
    with open(os.path.join(args.root, "summary.md"), "w") as f:
        f.write(md + "\n")
    figure(runs, os.path.join(args.root, "sweep.png"))


if __name__ == "__main__":
    main()
