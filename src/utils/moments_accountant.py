"""
Moments Accountant for the sampled Gaussian mechanism (Abadi et al., 2016).
"""

import math
import numpy as np
from scipy.special import logsumexp


def _gauss_log_pdf(x, mean, sigma):
    return -((x - mean) ** 2) / (2.0 * sigma ** 2) - math.log(math.sqrt(2 * math.pi) * sigma)


def _step_log_moment(q, sigma, lmbd):

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

    def __init__(self, sigma, delta, orders=None):
        self.sigma = sigma
        self.delta = delta
        if orders is None:
            # lambda = alpha - 1, with alpha from 1.1..9.9 (step 0.1) then 11..63
            orders = [round(0.1 * x, 4) for x in range(1, 99)] + list(range(10, 63))
        self.orders = orders
        self._log_moments = np.zeros(len(self.orders))  # cumulative mu(lambda)

    def step(self, q, num_steps=1):
        for i, lmbd in enumerate(self.orders):
            self._log_moments[i] += num_steps * _step_log_moment(q, self.sigma, lmbd)

    def get_epsilon(self):
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
