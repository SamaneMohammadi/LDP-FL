"""
Client side of LDP-FL (Algorithm 1, lines 6-11).

Each selected client runs DP-SGD over its local data and sends only the noisy
gradient to the server. Per local batch:

  1. true per-sample gradients (via torch.func, so clipping is exactly per-sample)
  2. clip each sample to L2 norm C
  3. sum, add Gaussian noise N(0, sigma^2 C^2 I), divide by the batch size

A round does one pass over the local data, so a client with more data runs more
noisy batch steps -> it spends more of its privacy budget per round. Each client
keeps its OWN Moments Accountant, so the cumulative epsilon reflects how much
that specific client actually participated (which CSS makes unequal).

The client never sends raw data or an un-noised gradient. It does, however,
report its dataset size to the server so CSS can rank clients (see report_size).
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

from dataset import get_loaders
from utils.moments_accountant import MomentsAccountant
import config


def set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(w)


class Client:
    def __init__(self, cid, model, sigma, delta, data_dir="client_data", device="cpu"):
        self.cid = cid
        self.model = model
        self.device = device
        self.sigma = sigma
        self.train_loader, self.val_loader, self.n_train = get_loaders(
            cid, data_dir=data_dir, batch_size=config.BATCH_SIZE
        )
        # this client's own privacy ledger, and its per-step sampling rate
        self.accountant = MomentsAccountant(sigma=sigma, delta=delta)
        self.q = min(1.0, config.BATCH_SIZE / max(self.n_train, 1))
        self.rounds_participated = 0

    def report_size(self):
        """What the client discloses to the server so CSS can rank it."""
        return self.n_train

    def _per_sample_grads(self, x, y):
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        def loss_on_one(params, xi, yi):
            out = functional_call(self.model, (params, buffers), (xi.unsqueeze(0),))
            return F.nll_loss(out, yi.unsqueeze(0))

        grads = vmap(grad(loss_on_one), in_dims=(None, 0, 0),
                     randomness="different")(params, x, y)
        return [grads[k] for k in params]

    def _noisy_batch_grad(self, x, y, clip):
        """One DP-SGD step on a batch: per-sample clip, sum, add noise, average."""
        n = len(x)
        per_sample = self._per_sample_grads(x, y)
        flat = torch.cat([g.reshape(n, -1) for g in per_sample], dim=1)
        scale = (clip / flat.norm(dim=1)).clamp(max=1.0)   # max(1, ||g||/C)^-1

        out = []
        for g in per_sample:
            g_clipped = g * scale.view(-1, *([1] * (g.dim() - 1)))
            summed = g_clipped.sum(dim=0)
            noise = torch.normal(0.0, self.sigma * clip, size=summed.shape, device=self.device)
            out.append((summed + noise) / n)
        return out

    def noisy_gradient(self, global_weights, clip):
        """DP-SGD over the local data; returns the round's noisy gradient.

        Also advances this client's accountant by the number of noisy batch
        steps it ran this round.
        """
        set_weights(self.model, global_weights)
        self.model.train()

        accum, n_batches = None, 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            bg = self._noisy_batch_grad(x, y, clip)
            accum = bg if accum is None else [a + b for a, b in zip(accum, bg)]
            n_batches += 1

        # account: each batch was one sub-sampled Gaussian release at rate q
        self.accountant.step(q=self.q, num_steps=n_batches)
        self.rounds_participated += 1

        return [a / n_batches for a in accum]

    def epsilon(self):
        """Cumulative privacy spent by THIS client so far."""
        return self.accountant.get_epsilon()
