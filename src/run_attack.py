"""
Model inversion attack evaluation (paper Table I).

Trains FL with K=7 clients per round, then attacks the local model each
selected client shares in the last round (global weights minus that client's
(noisy) gradient step) and reports the MSE between the reconstruction and the
client's real features for the target label, averaged over clients and labels.
"""

import argparse
import copy
import random

import numpy as np
import torch

import config
from attack import invert
from client import Client, set_weights
from dataset import _load, count_clients
from main import get_weights
from model import build_model
from server import Server
from utils.css import select


def run(sigma, clip, no_dp, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    global_model = build_model()
    server = Server()
    clients = [Client(cid, global_model, sigma=sigma, delta=args.delta,
                      data_dir=args.data_dir, dp=not no_dp)
               for cid in range(count_clients(args.data_dir))]
    sizes = [c.report_size() for c in clients]

    for rnd in range(1, args.rounds + 1):
        weights = get_weights(global_model)
        chosen = select(sizes, args.clients_per_round, args.selection, rng)
        grads = [clients[i].noisy_gradient(weights, clip) for i in chosen]
        if rnd == args.rounds:
            break
        set_weights(global_model, server.aggregate_and_update(weights, grads, config.LEARNING_RATE))

    # attack the local model each client shares in the final round
    mses = []
    for cid, g in zip(chosen, grads):
        local = copy.deepcopy(global_model)
        set_weights(local, [w - config.LEARNING_RATE * gi for w, gi in zip(weights, g)])
        x, y = _load(args.data_dir, cid, "train")
        for label in range(config.NUM_CLASSES):
            recon, _ = invert(local, label, input_dim=x.shape[1], clamp=not args.no_clamp)
            recon = recon.flatten().numpy()
            mses.extend(((xi - recon) ** 2).mean() for xi in x[y == label])
    return float(np.mean(mses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=float, nargs="+", default=config.CLIP_NORMS)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.0, 10.0])
    parser.add_argument("--delta", type=float, default=config.DEFAULT_DELTA)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients_per_round", type=int, default=7)
    parser.add_argument("--selection", choices=["css", "random"], default=config.SELECTION)
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no_clamp", action="store_true",
                        help="skip Process() clamping of the reconstruction to [0,1]")
    args = parser.parse_args()

    print(f"non-LDP-FL: MSE {run(0.0, 0.0, True, args):.3f}")
    for clip in args.clips:
        row = [f"{run(s, clip, False, args):.3f}" for s in args.sigmas]
        print(f"LDP-FL C={clip}: " + "  ".join(f"sigma={s}: {m}" for s, m in zip(args.sigmas, row)))


if __name__ == "__main__":
    main()
