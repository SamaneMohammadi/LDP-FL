"""
Plot per-client privacy spent, random selection vs CSS, from the CSVs main.py
writes: (a) epsilon vs local data size per client, (b) worst-case epsilon over rounds.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SERIES = {"random": ("Random selection", "#2a78d6"), "css": ("CSS", "#eb6834")}


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/skewed")
    parser.add_argument("--tag", default="sigma1.0_C2.0_delta1e-05_K50",
                        help="run tag after the selection name in the CSV file names")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sizes = []
    for sel, (label, color) in SERIES.items():
        clients = read_rows(os.path.join(args.results_dir, f"eps_clients_{sel}_{args.tag}.csv"))
        clients = [c for c in clients if int(c["rounds_participated"]) > 0]
        sizes += [int(c["n_train"]) for c in clients]
        ax1.scatter([int(c["n_train"]) for c in clients], [float(c["epsilon"]) for c in clients],
                    s=36, color=color, edgecolor="white", linewidth=1.5, label=label, zorder=3)

        rounds = read_rows(os.path.join(args.results_dir, f"eps_rounds_{sel}_{args.tag}.csv"))
        ax2.plot([int(r["round"]) for r in rounds], [float(r["eps_max"]) for r in rounds],
                 color=color, linewidth=2, label=label)
        ax2.annotate(label, (int(rounds[-1]["round"]), float(rounds[-1]["eps_max"])),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="#52514e")

    if max(sizes) / min(sizes) > 4:
        ax1.set_xscale("log")
        ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_xlabel("Client training samples (log scale)")
    else:
        ax1.set_xlabel("Client training samples")
    ax1.set_ylabel("Privacy budget spent, ε")
    ax1.set_title("(a) ε per client after training", loc="left", fontsize=11)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Worst-case client ε")
    ax2.set_title("(b) System guarantee over rounds", loc="left", fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(color="#e4e3df", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9b9a95")
        ax.tick_params(colors="#52514e")
    ax2.margins(x=0.12)
    title = (args.tag.replace("sigma", "σ=").replace("_C", ", C=").replace("_delta", ", δ=")
             .replace("_K", ", K="))
    fig.suptitle(f"Per-client privacy budget ({title})", x=0.01, ha="left", fontsize=12)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or os.path.join(args.results_dir, f"privacy_{args.tag}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
