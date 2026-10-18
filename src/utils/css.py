import random


def css_select(client_sizes, k, rng=random):
    """Client Selection Strategy (paper Algorithm 2).

    M = K/2 clients with the largest local datasets, plus K - M clients drawn
    uniformly at random from the remaining ones (no overlap). Clients with equal
    sample sizes are ordered randomly each round, so ties are not decided by
    client id.
    """
    n = len(client_sizes)
    k = min(k, n)
    m = k // 2  # half by size

    # clients sorted by sample size, largest first; random order among ties
    tiebreak = [rng.random() for _ in range(n)]
    by_size = sorted(range(n), key=lambda i: (-client_sizes[i], tiebreak[i]))
    top = by_size[:m]

    # the rest are drawn at random from everyone not already chosen
    rest = rng.sample(by_size[m:], k - m)

    return top + rest


def random_selection(client_sizes, k, rng=random):

    n = len(client_sizes)
    return rng.sample(range(n), min(k, n))


def select(client_sizes, k, strategy="css", rng=random):
    if strategy == "css":
        return css_select(client_sizes, k, rng)
    return random_selection(client_sizes, k, rng)
