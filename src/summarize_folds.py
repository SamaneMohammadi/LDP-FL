"""
Average a k-fold experiment over folds: test accuracy and per-client privacy,
random selection vs CSS, for every configuration.

Expects <root>/<config>/fold<k>/summary_<selection>_<mode>.json and
eps_clients_*.csv as written by main.py, e.g. results/folds/client_s3.0/fold0/.
Writes <root>/summary.md and <root>/eps_vs_size.png.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"random": ("Random selection", "#2a78d6"), "css": ("CSS", "#eb6834")}
SIZE_GROUPS = [(1, 12), (13, 22), (23, 33), (34, 10 ** 6)]


def group_name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def load(root):
    runs = defaultdict(list)       # (config, selection) -> [summary per fold]
    clients = defaultdict(list)    # (config, selection) -> [per-client rows over folds]
    for path in sorted(glob.glob(os.path.join(root, "*", "fold*", "summary_*.json"))):
        cfg = path.split(os.sep)[-3]
        sel = re.match(r"summary_(random|css)_", os.path.basename(path)).group(1)
        with open(path) as f:
            runs[(cfg, sel)].append(json.load(f))
        for c in glob.glob(os.path.join(os.path.dirname(path), f"eps_clients_{sel}_*.csv")):
            with open(c) as f:
                clients[(cfg, sel)] += [r for r in csv.DictReader(f) if int(r["rounds_participated"]) > 0]
    return runs, clients


def ms(vals, fmt="{:.1f}"):
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals))}"


def table(runs, clients):
    out = ["| Config | Selection | Test accuracy (%) | Worst-case ε | Median ε | "
           + " | ".join(f"ε, size {group_name(*g)}" for g in SIZE_GROUPS) + " | folds |",
           "|---|---|---|---|---|" + "---|" * len(SIZE_GROUPS) + "---|"]
    for cfg in sorted({c for c, _ in runs}):
        for sel in SERIES:
            rs = runs.get((cfg, sel))
            if not rs:
                continue
            assert all(r["evaluated_on"] == "test" for r in rs), "folds must be evaluated on test speakers"
            row = f"| {cfg} | {sel} | {ms([100 * r['acc_last3'] for r in rs])} |"
            if "privacy" in rs[0]:
                row += f" {ms([r['privacy']['eps_max'] for r in rs])} | {ms([r['privacy']['eps_median'] for r in rs])} |"
                cl = clients[(cfg, sel)]
                for lo, hi in SIZE_GROUPS:
                    g = [r for r in cl if lo <= int(r["n_train"]) <= hi]
                    row += (f" {np.mean([float(r['epsilon']) for r in g]):.1f} "
                            f"({np.mean([int(r['steps_per_round']) for r in g]):.1f} batches, "
                            f"{np.mean([int(r['rounds_participated']) for r in g]):.0f} rounds) |") if g else " – |"
            else:
                row += " – | – |" + " – |" * len(SIZE_GROUPS)
            out.append(row + f" {len(rs)} |")
    return "\n".join(out)


def figure(clients, configs, out):
    fig, axes = plt.subplots(1, len(configs), figsize=(5.5 * len(configs), 4.4), squeeze=False)
    for ax, (cfg, title) in zip(axes[0], configs):
        for sel, (label, color) in SERIES.items():
            cl = clients.get((cfg, sel), [])
            ax.scatter([int(r["n_train"]) for r in cl], [float(r["epsilon"]) for r in cl], s=22,
                       color=color, edgecolor="white", linewidth=1, alpha=0.85, label=label, zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Client training samples")
        ax.set_ylabel("Privacy budget spent, ε")
        ax.grid(color="#e4e3df", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9b9a95")
        ax.tick_params(colors="#52514e")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Per-client ε vs data size (all clients, all folds)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/folds")
    parser.add_argument("--plot", nargs="*", default=["record_s3.0", "client_s3.0"],
                        help="configurations to plot per-client epsilon for")
    args = parser.parse_args()
    runs, clients = load(args.root)
    md = table(runs, clients)
    print(md)
    with open(os.path.join(args.root, "summary.md"), "w") as f:
        f.write(md + "\n")
    titles = {"record_s3.0": "(a) Recording-level privacy, σ=3", "client_s3.0": "(b) Speaker-level privacy, σ=3"}
    figure(clients, [(c, titles.get(c, c)) for c in args.plot], os.path.join(args.root, "eps_vs_size.png"))


if __name__ == "__main__":
    main()
