from itertools import product
import numpy as np
import pytest
from research.trajectory_movie_routing_bound_v1 import fractional_bound


def test_matches_exhaustive_discrete_choices():
    rng = np.random.default_rng(20260914)
    for _ in range(20):
        n = rng.integers(0, 100, size=(6, 2))
        d = n + rng.integers(1, 100, size=(6, 2))
        exact = max(sum(n[i, c] for i, c in enumerate(choices)) /
                    sum(d[i, c] for i, c in enumerate(choices)) for choices in product(range(2), repeat=6))
        assert fractional_bound(n, d)['ratio'] == pytest.approx(exact, abs=1e-14)


def test_individual_movie_ratio_maximum_is_not_pooled_optimum():
    n, d = np.array([[9, 80], [1, 1]]), np.array([[10, 100], [100, 100]])
    assert n[0, 0] / d[0, 0] > n[0, 1] / d[0, 1]
    assert fractional_bound(n, d)['ratio'] == pytest.approx(81 / 200)


@pytest.mark.parametrize('n,d', [([], []), ([[1]], [[0]]), ([[float('nan')]], [[1]])])
def test_invalid_inputs_rejected(n, d):
    with pytest.raises(ValueError):
        fractional_bound(n, d)
