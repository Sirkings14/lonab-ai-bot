import numpy as np
from ticket_math import (race_weights, top_k_marginals, prob_overlap_at_least,
                         prob_exact_set, prob_exact_order, iter_prefixes)

W = race_weights([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05])

def test_weights_normalised():
    assert abs(W.sum() - 1) < 1e-12

def test_prefix_probabilities_sum_to_one():
    assert abs(sum(p for _, p in iter_prefixes(W, 3)) - 1) < 1e-9

def test_marginals_sum_to_k():
    assert abs(top_k_marginals(W, 3).sum() - 3) < 1e-9

def test_favourite_more_likely_top3():
    m = top_k_marginals(W, 3)
    assert m[0] > m[1] > m[-1]

def test_exact_set_is_sum_of_orders_and_beats_exact_order():
    s, o = prob_exact_set(W, [0, 1, 2]), prob_exact_order(W, [0, 1, 2])
    assert s > o and abs(s - prob_overlap_at_least(W, [0, 1, 2], 3, 3)) < 1e-9

def test_overlap_monotone():
    a = [prob_overlap_at_least(W, [0, 1, 2], 3, m) for m in (0, 1, 2, 3)]
    assert abs(a[0] - 1) < 1e-9 and a[0] >= a[1] >= a[2] >= a[3]

def test_matches_simulation():
    rs = np.random.RandomState(0); n = 40000; hit = 0
    for _ in range(n):
        order = rs.choice(len(W), 3, replace=False, p=W)
        hit += len(set(order) & {0, 1, 2}) >= 2
    assert abs(hit / n - prob_overlap_at_least(W, [0, 1, 2], 3, 2)) < 0.01
