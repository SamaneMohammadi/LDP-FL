import argparse
import csv
import json
import random

import numpy as np
import torch

import config
from model import build_model
from client import Client, set_weights
from server import Server
from utils.css import select
from utils.metrics import evaluate
from dataset import FeatureDataset, count_clients, get_test_loader
from torch.utils.data import DataLoader
import os


def get_weights(model):
    return [p.detach().clone() for p in model.parameters()]


def build_global_val_loader(data_dir, num_clients):
    xs, ys = [], []
    for cid in range(num_clients):
        xp = os.path.join(data_dir, f"client_{cid}_x_val.npy")
        if os.path.exists(xp):
            xs.append(np.load(xp))
            ys.append(np.load(os.path.join(data_dir, f"client_{cid}_y_val.npy")))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return DataLoader(FeatureDataset(x, y), batch_size=256, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=["css", "random"], default=config.SELECTION)
    parser.add_argument("--sigma", type=float, default=config.DEFAULT_SIGMA)
    parser.add_argument("--clip", type=float, default=config.DEFAULT_CLIP)
    parser.add_argument("--delta", type=float, default=config.DEFAULT_DELTA)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--clients_per_round", type=int, default=config.CLIENTS_PER_ROUND)
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--no_dp", action="store_true",
                        help="non-LDP baseline: no clipping, no noise")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--lot_frac", type=float, default=None,
                        help="same sampling rate q for every client (lot = lot_frac * n); "
                             "default: lot of B samples, q = B/n")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="server learning rate eta of the FedSGD update")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="expected local minibatch size B")
    parser.add_argument("--local_epoch", action="store_true",
                        help="each selected client runs one local epoch of noisy steps per "
                             "round instead of the single FedSGD step of Algorithm 1")
    parser.add_argument("--privacy_unit", choices=["record", "client"], default="record",
                        help="record: protect one recording (per-sample clipping); client: "
                             "protect a client's whole dataset (clip each local step's "
                             "batch gradient), so more local steps cost more privacy")
    parser.add_argument("--out_dir", default="results",
                        help="where the per-client epsilon CSVs are written")
    args = parser.parse_args()

    config.BATCH_SIZE = args.batch_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cpu"

    global_model = build_model().to(device)
    num_clients = count_clients(args.data_dir)
    val_loader = build_global_val_loader(args.data_dir, num_clients)
    # accuracy is reported on held-out test speakers when the data has them
    # (prepare_data.py --folds); otherwise on the clients' pooled validation split
    test_loader = get_test_loader(args.data_dir)
    server = Server()

    clients = [Client(cid, global_model, sigma=args.sigma, delta=args.delta,
                      data_dir=args.data_dir, device=device, dp=not args.no_dp,
                      lot_frac=args.lot_frac, local_epoch=args.local_epoch,
                      privacy_unit=args.privacy_unit)
               for cid in range(num_clients)]
    # clients disclose their dataset size to the server so CSS can rank them
    client_sizes = [c.report_size() for c in clients]

    mode = "non-LDP FL" if args.no_dp else "LDP-FL"
    print(f"{mode} | selection={args.selection} sigma={args.sigma} C={args.clip} "
          f"delta={args.delta} K={args.clients_per_round} B={args.batch_size} "
          f"privacy_unit={args.privacy_unit} local_epoch={args.local_epoch} | "
          f"{num_clients} clients, evaluated on "
          f"{'held-out test speakers' if test_loader else 'pooled validation split'}")

    history = []
    for rnd in range(1, args.rounds + 1):
        weights = get_weights(global_model)
        chosen = select(client_sizes, args.clients_per_round, args.selection, rng)

        grads = [clients[i].noisy_gradient(weights, args.clip, rnd) for i in chosen]
        new_weights = server.aggregate_and_update(weights, grads, args.lr)
        set_weights(global_model, new_weights)

        if rnd % args.eval_every == 0 or rnd == args.rounds:
            v = evaluate(global_model, val_loader, device)
            m = evaluate(global_model, test_loader, device) if test_loader else v
            history.append({"round": rnd, **m, "val_acc": v["acc"]})
            line = (f"round {rnd:3d} | {'test' if test_loader else 'val'} acc {m['acc']:.4f} "
                    f"f1 {m['f1']:.4f} prec {m['precision']:.4f} loss {m['loss']:.4f}")
            if not args.no_dp:
                # the LDP guarantee must hold for every client -> worst case
                eps_max = max((c.epsilon() for c in clients if c.rounds_participated), default=0.0)
                line += f" | eps_max {eps_max:.2f}"
            print(line, flush=True)

    summary = {"args": vars(args), "history": history, "final": history[-1],
               "evaluated_on": "test" if test_loader else "val",
               # mean of the last three evaluations, less sensitive to one noisy round
               "acc_last3": float(np.mean([h["acc"] for h in history[-3:]]))}
    if not args.no_dp:
        summary["privacy"] = report_privacy(clients, args)
    os.makedirs(args.out_dir, exist_ok=True)
    mode = "nondp" if args.no_dp else f"sigma{args.sigma}"
    with open(os.path.join(args.out_dir, f"summary_{args.selection}_{mode}.json"), "w") as f:
        json.dump(summary, f, indent=1)


def report_privacy(clients, args):
    """Per-client privacy spent. Each client's epsilon depends on its own data
    size (q = B/n, n/B steps per round) and on how often it was selected, so
    under CSS the large clients spend far more. The system-level (eps, delta)-LDP
    guarantee is the worst case over clients, not an average or a single client."""
    rows = [(c.cid, c.n_train, c.acct_q, c.steps_per_round, c.rounds_participated,
             c.epsilon() if c.rounds_participated else 0.0) for c in clients]
    eps_vals = sorted(r[5] for r in rows)
    part = [r for r in rows if r[4] > 0]

    print(f"\nper-client {args.privacy_unit}-level privacy (delta={args.delta}, "
          f"{len(part)}/{len(rows)} clients participated):")
    print(f"  system guarantee: ({eps_vals[-1]:.2f}, {args.delta})-LDP  [worst-case client]")
    print(f"  epsilon over clients: min {eps_vals[0]:.2f}  median {eps_vals[len(eps_vals) // 2]:.2f}  "
          f"max {eps_vals[-1]:.2f}")

    if args.selection == "css":
        # CSS always picks the K/2 largest clients; the rest are picked at random
        m = min(args.clients_per_round, len(clients)) // 2
        top = set(sorted(range(len(clients)), key=lambda i: clients[i].n_train, reverse=True)[:m])
        for name, grp in [("CSS size-selected", [r for r in rows if r[0] in top]),
                          ("random pool", [r for r in rows if r[0] not in top])]:
            e = [r[5] for r in grp]
            print(f"  {name:18s} ({len(grp):2d} clients): rounds {min(r[4] for r in grp)}-"
                  f"{max(r[4] for r in grp)} | epsilon {min(e):.2f}-{max(e):.2f}")

    print("  heaviest spenders (cid, data_size, q, steps/round, rounds, epsilon):")
    for cid, n, q, spr, r, e in sorted(rows, key=lambda r: r[5], reverse=True)[:5]:
        print(f"    client {cid:3d} | size {n:4d} | q {q:.3f} | steps {spr} | rounds {r:3d} | eps {e:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.selection}_sigma{args.sigma}_C{args.clip}_delta{args.delta}_K{args.clients_per_round}"
    with open(os.path.join(args.out_dir, f"eps_clients_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", "n_train", "q", "steps_per_round", "rounds_participated", "epsilon"])
        w.writerows(rows)
    # epsilon trajectory per client (paper Fig. 4), plus the worst case per round
    with open(os.path.join(args.out_dir, f"eps_rounds_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "eps_max"] + [f"client_{c.cid}" for c in clients])
        current = [0.0] * len(clients)
        hist = {c.cid: dict(c.eps_history) for c in clients}
        for rnd in range(1, args.rounds + 1):
            for i, c in enumerate(clients):
                current[i] = hist[c.cid].get(rnd, current[i])
            w.writerow([rnd, max(current)] + [f"{e:.4f}" for e in current])
    print(f"  wrote {args.out_dir}/eps_clients_{tag}.csv and eps_rounds_{tag}.csv")

    part = [r[5] for r in rows if r[4] > 0]
    return {"eps_max": eps_vals[-1], "eps_median": float(np.median(part)),
            "eps_min": float(min(part)),
            "eps_size_corr": float(np.corrcoef([r[1] for r in rows if r[4] > 0], part)[0, 1])}

if __name__ == "__main__":
    main()
