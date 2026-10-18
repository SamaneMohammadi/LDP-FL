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
    def __init__(self, cid, model, sigma, delta, data_dir="client_data", device="cpu", dp=True,
                 lot_frac=None, local_epoch=False, privacy_unit="record"):
        self.cid = cid
        self.model = model
        self.device = device
        self.sigma = sigma
        self.dp = dp
        self.privacy_unit = privacy_unit
        self.train_loader, self.val_loader, self.n_train = get_loaders(
            cid, data_dir=data_dir, batch_size=config.BATCH_SIZE
        )
        # this client's own privacy ledger. Each noisy step Poisson-samples a lot
        # with rate q -- exactly what the accountant assumes. By default the
        # expected lot is B samples (q = B/n, so q depends on the client's size);
        # with lot_frac every client uses the same q = lot_frac (lot = lot_frac * n).
        # Per round a selected client takes ONE noisy step (paper Algorithm 1,
        # FedSGD); with local_epoch it takes 1/q steps (one pass in expectation).
        self.accountant = MomentsAccountant(sigma=sigma, delta=delta)
        if lot_frac is None:
            self.q = min(1.0, config.BATCH_SIZE / max(self.n_train, 1))
            epoch_steps = max(1, round(self.n_train / config.BATCH_SIZE))
        else:
            self.q = lot_frac
            epoch_steps = max(1, round(1.0 / lot_frac))
        self.steps_per_round = epoch_steps if local_epoch else 1
        self.lot_size = self.q * self.n_train
        # Privacy unit. "record": protects one recording; each step is a Gaussian
        # release sub-sampled at rate q. "client": protects the client's whole
        # dataset (the speaker); every step touches that dataset, so there is no
        # sub-sampling amplification (accounting rate 1) and each extra local step
        # costs privacy -- clients running more steps per round spend more.
        self.acct_q = self.q if privacy_unit == "record" else 1.0
        self.x_train = self.train_loader.dataset.x.to(device)
        self.y_train = self.train_loader.dataset.y.to(device)
        self.rounds_participated = 0
        self.eps_history = []   # (round, cumulative epsilon) after each participation

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

    def _batch_grad(self, x, y):
        """Plain (non-private) mean gradient of the batch, for the non-LDP baseline."""
        params = list(self.model.parameters())
        loss = F.nll_loss(self.model(x), y)
        return [g.detach() for g in torch.autograd.grad(loss, params)]

    def _noisy_lot_grad(self, clip):
        """One DP-SGD step: Poisson-sample a lot, per-sample clip, sum, add noise,
        divide by the expected lot size (Abadi et al., 2016)."""
        mask = torch.rand(self.n_train, device=self.device) < self.q
        x, y = self.x_train[mask], self.y_train[mask]
        params = [p for p in self.model.parameters()]

        if len(x) > 0:
            n = len(x)
            per_sample = self._per_sample_grads(x, y)
            flat = torch.cat([g.reshape(n, -1) for g in per_sample], dim=1)
            scale = (clip / flat.norm(dim=1).clamp(min=1e-12)).clamp(max=1.0)   # max(1, ||g||/C)^-1
            sums = [(g * scale.view(-1, *([1] * (g.dim() - 1)))).sum(dim=0) for g in per_sample]
        else:
            sums = [torch.zeros_like(p) for p in params]   # empty lot: release noise only

        out = []
        for summed in sums:
            noise = torch.normal(0.0, self.sigma * clip, size=summed.shape, device=self.device)
            out.append((summed + noise) / self.lot_size)
        return out

    def _noisy_client_grad(self, clip):
        """One client-level DP step: sample a minibatch, clip the client's whole
        batch gradient to C (sensitivity C w.r.t. the entire client dataset) and
        add N(0, sigma^2 C^2 I)."""
        mask = torch.rand(self.n_train, device=self.device) < self.q
        if not mask.any():
            mask[torch.randint(self.n_train, (1,))] = True
        g = self._batch_grad(self.x_train[mask], self.y_train[mask])
        norm = torch.sqrt(sum((gi ** 2).sum() for gi in g))
        scale = (clip / norm.clamp(min=1e-12)).clamp(max=1.0)
        return [gi * scale + torch.normal(0.0, self.sigma * clip, size=gi.shape, device=self.device)
                for gi in g]

    def noisy_gradient(self, global_weights, clip, rnd=None):
        """DP-SGD over the local data; returns the round's noisy gradient.

        Also advances this client's accountant by the number of noisy steps it
        ran this round and records its cumulative epsilon.
        """
        set_weights(self.model, global_weights)
        self.model.train()

        if self.dp:
            step_fn = self._noisy_lot_grad if self.privacy_unit == "record" else self._noisy_client_grad
            steps = [step_fn(clip) for _ in range(self.steps_per_round)]
            # account: each step is one Gaussian release (sub-sampled at rate
            # acct_q for record-level privacy, not sub-sampled for client-level)
            self.accountant.step(q=self.acct_q, num_steps=self.steps_per_round)
        else:
            # non-LDP baseline: the same steps on the same lot sizes, without
            # clipping or noise
            steps = []
            for _ in range(self.steps_per_round):
                mask = torch.rand(self.n_train, device=self.device) < self.q
                if not mask.any():
                    mask[torch.randint(self.n_train, (1,))] = True
                steps.append(self._batch_grad(self.x_train[mask], self.y_train[mask]))
        self.rounds_participated += 1
        if self.dp:
            self.eps_history.append((rnd, self.epsilon()))

        return [torch.stack(gs).mean(dim=0) for gs in zip(*steps)]

    def epsilon(self):
        """Cumulative privacy spent by THIS client so far."""
        if not self.dp:
            return float("inf")
        return self.accountant.get_epsilon()
