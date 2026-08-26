from __future__ import annotations

import numpy as np

from research.hoct_graph.train_biohub_hoct_probe import (
    covering_window_starts,
    select_probe_examples,
)


def test_covering_windows_include_every_consecutive_transition() -> None:
    times = np.repeat(np.arange(13), 2)
    starts = covering_window_starts(times, window_size=5, stride=4)
    covered = {
        t
        for start in starts
        for t in range(start, start + 4)
        if 0 <= t < 12
    }
    assert covered == set(range(12))


def test_probe_sampling_keeps_positives_and_hard_negatives() -> None:
    features = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    logits = np.asarray([0, 0, 10, 9, 8, 7, 6, 5], dtype=np.float32)
    selected_features, selected_labels = select_probe_examples(
        features,
        labels,
        logits,
        negative_ratio=2,
        hard_negative_fraction=0.5,
        rng=np.random.default_rng(11),
    )

    selected_rows = {int(row[0] // 3) for row in selected_features}
    assert {0, 1, 2, 3}.issubset(selected_rows)
    assert selected_labels.sum() == 2
    assert len(selected_labels) == 6
