from __future__ import annotations

import numpy as np

from research.train_handcrafted_division_gate import (
    high_precision_metrics,
    patch_features,
)


def test_handcrafted_feature_contract_is_finite_and_temporal() -> None:
    patches = np.zeros((2, 3, 17, 17, 17), dtype=np.float32)
    patches[0, 2, 8, 5, 8] = 4.0
    patches[0, 2, 8, 11, 8] = 3.5
    patches[1, 1, 8, 8, 8] = 4.0

    features = patch_features(patches)

    assert features.shape == (2, 132)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])


def test_high_precision_metrics_reward_clean_top_ranking() -> None:
    targets = np.asarray([1, 1, 0, 0])
    clean = high_precision_metrics(targets, np.asarray([0.9, 0.8, 0.2, 0.1]))
    noisy = high_precision_metrics(targets, np.asarray([0.9, 0.1, 0.8, 0.2]))

    assert clean["average_precision"] > noisy["average_precision"]
    assert clean["recall_at_zero_false_positives"] == 1.0
    assert noisy["recall_at_zero_false_positives"] == 0.5
