"""
Moments Accountant for the sampled Gaussian mechanism (Abadi et al., 2016).

The paper tracks cumulative privacy loss with the Moments Accountant rather than
plain composition, because it gives much tighter (epsilon, delta) bounds under
the kind of repeated, sub-sampled Gaussian noise that DP-SGD produces. Opacus
ships an RDP accountant but does not expose the log-moment formulation directly,
so this is a small standalone implementation.

For each step we compute the log moment

    mu_step(lambda) = log E[ (M(D)/M(D'))^lambda ]

of the sub-sampled Gaussian mechanism with sampling rate q and noise multiplier
sigma. Moments compose additively, so after T steps

    mu(lambda) = T * mu_step(lambda)

and the privacy budget is

    epsilon = min_lambda ( mu(lambda) - log(delta) ) / lambda      (delta fixed).

This matches Eq. (Moments Accountant) and Section III "Privacy Loss Accounting".
"""

import math
import numpy as np
from scipy.special import logsumexp


def _gauss_log_pdf(x, mean, sigma):
    return -((x - mean) ** 2) / (2.0 * sigma ** 2) - math.log(math.sqrt(2 * math.pi) * sigma)


def _step_log_moment(q, sigma, lmbd):
    """Log moment of one sub-sampled Gaussian step, for moment order lmbd.

    mu0 ~ N(0, sigma^2) is the "data absent" distribution; the mechanism output
    mixes mu0 and mu1 ~ N(1, sigma^2) with weight q. We take the larger of the
    two directional moments, as in the original accountant. Everything is done
    in log-space to stay stable in the tails (otherwise (mix/mu0)^lambda
    overflows for large lambda).

    Integration window: the second integrand, mix*(mix/mu0)^lmbd, peaks at
    z ~= 1 + lmbd (its log-derivative is (1 + lmbd - z)/sigma^2), so the upper
    limit MUST grow with lmbd or the peak is truncated and the moment is
    under-estimated. We extend to 1 + lmbd + 12*sigma and pick the grid spacing
    fine enough (<= sigma/30) to resolve the ~sigma-wide peak.
    """
    lo = -1.0 - 12.0 * sigma
    hi = 1.0 + lmbd + 12.0 * sigma
    grid_points = max(40000, int(30.0 * (hi - lo) / sigma))
    x = np.linspace(lo, hi, grid_points)
    log_dx = math.log((hi - lo) / (grid_points - 1))

    log_p0 = -(x ** 2) / (2.0 * sigma ** 2) - math.log(math.sqrt(2 * math.pi) * sigma)
    log_p1 = -((x - 1.0) ** 2) / (2.0 * sigma ** 2) - math.log(math.sqrt(2 * math.pi) * sigma)
    log_1mq = math.log(1 - q) if q < 1 else -np.inf
    log_q = math.log(q) if q > 0 else -np.inf
    log_mix = np.logaddexp(log_1mq + log_p0, log_q + log_p1)

    # log E_{mu0}[ (mu0/mix)^lmbd ]  and  log E_{mix}[ (mix/mu0)^lmbd ]
    log_i0 = logsumexp(log_p0 + lmbd * (log_p0 - log_mix) + log_dx)
    log_i1 = logsumexp(log_mix + lmbd * (log_mix - log_p0) + log_dx)
    return max(log_i0, log_i1)


class MomentsAccountant:
    """Tracks cumulative log moments for one client across rounds.

    We minimise epsilon over a grid of moment orders lambda. The orders must
    include FRACTIONAL values below 1, not just integers: in the regime this
    paper runs in (q ~= 0.136, sigma as small as 0.5) the tightest bound is
    achieved at a small fractional order, and an integer-only grid would
    noticeably over-estimate epsilon. The grid below mirrors the orders used by
    Opacus' RDP accountant (expressed here as lambda = alpha - 1).
    """

    def __init__(self, sigma, delta, orders=None):
        self.sigma = sigma
        self.delta = delta
        if orders is None:
            # lambda = alpha - 1, with alpha from 1.1..9.9 (step 0.1) then 11..63
            orders = [round(0.1 * x, 4) for x in range(1, 99)] + list(range(10, 63))
        self.orders = orders
        self._log_moments = np.zeros(len(self.orders))  # cumulative mu(lambda)

    def step(self, q, num_steps=1):
        """Account for `num_steps` DP-SGD updates with sampling rate q."""
        for i, lmbd in enumerate(self.orders):
            self._log_moments[i] += num_steps * _step_log_moment(q, self.sigma, lmbd)

    def get_epsilon(self):
        """Current cumulative epsilon at the fixed delta.

        epsilon = min_lambda ( mu(lambda) - log(delta) ) / lambda
        """
        eps = [
            (m - math.log(self.delta)) / lmbd
            for m, lmbd in zip(self._log_moments, self.orders)
        ]
        return float(min(eps))


if __name__ == "__main__":
    # quick sanity check: epsilon should grow with rounds and shrink with sigma
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        acc = MomentsAccountant(sigma=sigma, delta=1e-5)
        acc.step(q=0.136, num_steps=60)
        print(f"sigma={sigma}  ->  epsilon={acc.get_epsilon():.3f}")
