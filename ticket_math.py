"""
Race maths for the bot: turns the odds into honest ticket probabilities.

Model: Plackett-Luce ("each place goes to a horse with chance proportional to its strength").
Strength = (odds-implied probability, normalised inside the race) ** POWER.

POWER = 1.08 was fitted on 305 stored races. Re-fit it whenever the database grows.
"""
import itertools
import numpy as np

POWER = 1.08


def race_weights(market_prob, power=POWER):
    """Per-horse strengths summing to 1. market_prob = 1/decimal odds (any scale; normalised here)."""
    q = np.asarray(market_prob, dtype=float)
    q = np.clip(q, 1e-6, None)
    q = q / q.sum()
    w = q ** power
    return w / w.sum()


def iter_prefixes(w, k):
    """Yield (ordered tuple of the first k finishers, probability) for every possible top-k order."""
    n = len(w)

    def rec(prefix, used, remaining, prob):
        if len(prefix) == k:
            yield tuple(prefix), prob
            return
        for i in range(n):
            if used & (1 << i):
                continue
            p = prob * w[i] / remaining
            prefix.append(i)
            yield from rec(prefix, used | (1 << i), remaining - w[i], p)
            prefix.pop()

    yield from rec([], 0, 1.0, 1.0)


def top_k_marginals(w, k=3):
    """P(horse i finishes in the first k) for every horse. Sums to k."""
    m = np.zeros(len(w))
    for prefix, p in iter_prefixes(w, k):
        for i in prefix:
            m[i] += p
    return m


def prob_overlap_at_least(w, picks, k, m):
    """P(at least m of `picks` are among the first k finishers)."""
    picks = set(picks)
    tot = 0.0
    for prefix, p in iter_prefixes(w, k):
        if len(picks & set(prefix)) >= m:
            tot += p
    return tot


def prob_exact_set(w, picks):
    """P(the first len(picks) finishers are exactly `picks`, in any order) - 'desordre' ticket."""
    return sum(_order_prob(w, perm) for perm in itertools.permutations(picks))


def prob_exact_order(w, picks):
    """P(first len(picks) finishers are `picks` in exactly this order) - 'ordre' ticket."""
    return _order_prob(w, picks)


def _order_prob(w, order):
    rem, pr = 1.0, 1.0
    for i in order:
        pr *= w[i] / rem
        rem -= w[i]
    return pr


def break_even_multiple(p):
    """A ticket that hits with probability p needs to pay at least 1/p times its stake to break even."""
    return float("inf") if p <= 0 else 1.0 / p
