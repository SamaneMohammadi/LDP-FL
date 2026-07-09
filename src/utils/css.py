import random


def css_select(client_sizes, k, rng=random):

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

    n = len(client_sizes)
    return rng.sample(range(n), min(k, n))


def select(client_sizes, k, strategy="css", rng=random):
    if strategy == "css":
        return css_select(client_sizes, k, rng)
    return random_selection(client_sizes, k, rng)
