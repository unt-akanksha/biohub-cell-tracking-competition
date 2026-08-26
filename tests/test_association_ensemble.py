from __future__ import annotations

import numpy as np
import pytest

from research.association_ensemble import blend_pair_scores


def _scores(source_ids, target_ids, matrix):
    return {
        0: (
            np.asarray(source_ids, dtype=np.int64),
            np.asarray(target_ids, dtype=np.int64),
            np.asarray(matrix, dtype=np.float32),
        )
    }


def test_blend_aligns_reordered_stable_ids() -> None:
    trackastra = _scores([10, 11], [20, 21], [[0.8, 0.2], [0.3, 0.7]])
    hoct = _scores([11, 10], [21, 20], [[0.6, 0.4], [0.1, 0.9]])
    blended = blend_pair_scores(trackastra, hoct, trackastra_weight=0.0)
    source_ids, target_ids, matrix = blended[0]

    np.testing.assert_array_equal(source_ids, [10, 11])
    np.testing.assert_array_equal(target_ids, [20, 21])
    np.testing.assert_allclose(matrix, [[0.9, 0.1], [0.4, 0.6]], atol=1e-6)


def test_logit_blend_preserves_identical_probabilities() -> None:
    scores = _scores([1], [2, 3], [[0.02, 0.91]])
    blended = blend_pair_scores(scores, scores, trackastra_weight=0.37)
    np.testing.assert_allclose(blended[0][2], scores[0][2], atol=1e-6)


def test_blend_rejects_mismatched_node_sets() -> None:
    first = _scores([1], [2], [[0.5]])
    second = _scores([1], [3], [[0.5]])
    with pytest.raises(ValueError, match="target-node sets differ"):
        blend_pair_scores(first, second, trackastra_weight=0.5)
