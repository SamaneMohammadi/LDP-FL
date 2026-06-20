"""
Client Selection Strategy (CSS) - Algorithm 2 in the paper.

The idea: noise from LDP hurts accuracy most when updates come from small, noisy
local datasets. So each round we deliberately pick half of the K clients to be
the ones with the *largest* datasets (more data -> more representative update,
better signal-to-noise), and pick the other half uniformly at random (to avoid
always ignoring the small clients and to keep some diversity).

  - top half  : the M = K/2 clients with the most local samples
  - rest half : K - M clients drawn at random from everyone else
  - the two sets never overlap, so exactly K distinct clients are returned

`random_selection` is the baseline the paper compares against.
"""

import random


def css_select(client_sizes, k, rng=random):
    """Select k clients: half by largest dataset, half random, no overlap.

    client_sizes -- list/array where client_sizes[i] is client i's sample count
    k            -- number of clients to select this round
    Returns a list of k distinct client indices.
    """
    n = len(client_sizes)
    k = min(k, n)
    m = k // 2  # half by size

    # clients sorted by sample size, largest first
    by_size = sorted(range(n), key=lambda i: client_sizes[i], reverse=True)
    top = by_size[:m]

    # the rest are drawn at random from everyone not already chosen
    remaining_pool = [i for i in by_size if i not in set(top)]
    rest = rng.sample(remaining_pool, k - m)

    return top + rest


def random_selection(client_sizes, k, rng=random):
    """Baseline: pick k clients uniformly at random (paper's RS comparison)."""
    n = len(client_sizes)
    return rng.sample(range(n), min(k, n))


def select(client_sizes, k, strategy="css", rng=random):
    if strategy == "css":
        return css_select(client_sizes, k, rng)
    return random_selection(client_sizes, k, rng)
