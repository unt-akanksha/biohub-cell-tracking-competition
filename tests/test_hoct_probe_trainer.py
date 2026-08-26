from __future__ import annotations

import numpy as np

from research.hoct_graph.train_biohub_hoct_probe import (
    _selection_configurations,
    balanced_cap_examples,
    covering_window_starts,
    robust_selection_key,
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


def test_balanced_cap_bounds_dense_movie_without_erasing_positives() -> None:
    features = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    labels = np.asarray([1] * 40 + [0] * 60, dtype=np.float32)

    capped_features, capped_labels = balanced_cap_examples(
        features,
        labels,
        max_examples=20,
        negative_ratio=3.0,
        rng=np.random.default_rng(17),
    )

    assert capped_features.shape == (20, 3)
    assert capped_labels.sum() == 5
    assert (capped_labels == 0).sum() == 15
    assert len({tuple(row) for row in capped_features.tolist()}) == 20


def test_balanced_cap_uses_spare_capacity_when_negatives_are_rare() -> None:
    features = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    labels = np.asarray([1] * 10 + [0] * 2, dtype=np.float32)

    _capped_features, capped_labels = balanced_cap_examples(
        features,
        labels,
        max_examples=8,
        negative_ratio=3.0,
        rng=np.random.default_rng(3),
    )

    assert len(capped_labels) == 8
    assert capped_labels.sum() == 6


def test_single_backbone_selection_prioritizes_transfer_between_embryos() -> None:
    concentrated = {
        "selection_min_delta_vs_base": -0.03,
        "selection_summary": {
            "proxy_score": 1.15,
            "worst_movie": 0.72,
            "div_fp": 0,
        },
        "method": "hoct_only",
    }
    transferable = {
        "selection_min_delta_vs_base": 0.002,
        "selection_summary": {
            "proxy_score": 0.96,
            "worst_movie": 0.91,
            "div_fp": 1,
        },
        "method": "raw_confidence_hybrid",
    }

    assert robust_selection_key(transferable) > robust_selection_key(concentrated)


def test_linker_grid_can_preserve_postprocess_fallback_divisions() -> None:
    configurations = _selection_configurations()

    conservative = [
        row
        for row in configurations
        if row["method"] == "raw_confidence_hybrid"
        and row["base_division_keep_probability"] == 0.80
    ]
    assert conservative
    assert any(
        row["base_lock_probability"] == 0.90
        and row["base_bonus"] == 0.10
        and row["edge_threshold"] == 0.10
        for row in conservative
    )
