"""
LDP-FL with CSS (Algorithm 1 + Algorithm 2).

Each round:
  - CSS picks K clients (half largest by data size, half random)
  - each selected client computes a clipped + noised local gradient (LDP)
  - the server averages them and steps the global model (FedSGD)

Privacy (epsilon) for a given noise scale is reported with the Moments
Accountant at the end.

    python main.py --selection css    --sigma 3 --clip 2 --rounds 200
    python main.py --selection random --sigma 3 --clip 2 --rounds 200   # baseline
"""

import argparse
import random

import numpy as np
import torch
from pytorch_lightning import seed_everything

import config
from model import build_model
from client import Client, set_weights
from server import Server
from utils.css import select
from utils.metrics import evaluate
from dataset import FeatureDataset
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
    args = parser.parse_args()

    seed_everything(config.SEED, workers=True)
    rng = random.Random(config.SEED)
    device = "cpu"

    global_model = build_model().to(device)
    val_loader = build_global_val_loader(args.data_dir, config.NUM_CLIENTS)
    server = Server()

    clients = [Client(cid, global_model, sigma=args.sigma, delta=args.delta,
                      data_dir=args.data_dir, device=device)
               for cid in range(config.NUM_CLIENTS)]
    # clients disclose their dataset size to the server so CSS can rank them
    client_sizes = [c.report_size() for c in clients]

    print(f"LDP-FL | selection={args.selection} sigma={args.sigma} C={args.clip} "
          f"delta={args.delta} K={args.clients_per_round}")

    for rnd in range(1, args.rounds + 1):
        weights = get_weights(global_model)
        chosen = select(client_sizes, args.clients_per_round, args.selection, rng)

        grads = [clients[i].noisy_gradient(weights, args.clip) for i in chosen]
        new_weights = server.aggregate_and_update(weights, grads, config.LEARNING_RATE)
        set_weights(global_model, new_weights)

        if rnd % args.eval_every == 0 or rnd == args.rounds:
            m = evaluate(global_model, val_loader, device)
            print(f"round {rnd:3d} | acc {m['acc']:.4f} f1 {m['f1']:.4f} "
                  f"prec {m['precision']:.4f} loss {m['loss']:.4f}")

    # per-client privacy spent: CSS favours big clients, so they participate more
    # and spend more budget. Report the distribution, not a single number.
    participants = [c for c in clients if c.rounds_participated > 0]
    eps = [(c.cid, c.n_train, c.rounds_participated, c.epsilon()) for c in participants]
    eps.sort(key=lambda r: r[3], reverse=True)
    eps_vals = [e[3] for e in eps]
    print(f"\nper-client privacy (delta={args.delta}):")
    print(f"  {len(participants)} clients participated | "
          f"epsilon min {min(eps_vals):.2f}  median {sorted(eps_vals)[len(eps_vals)//2]:.2f}  "
          f"max {max(eps_vals):.2f}")
    print("  heaviest spenders (cid, data_size, rounds, epsilon):")
    for cid, n, r, e in eps[:5]:
        print(f"    client {cid:3d} | size {n:4d} | rounds {r:3d} | eps {e:.2f}")


if __name__ == "__main__":
    main()
