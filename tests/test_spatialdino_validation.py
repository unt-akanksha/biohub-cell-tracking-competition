from __future__ import annotations

import numpy as np
import pytest

from research.spatialdino_association.validate_appearance_correction import (
    normalize_frame,
    robust_selection_key,
    selection_configurations,
)


def test_normalize_frame_is_bounded_and_rejects_constant_input() -> None:
    frame = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    normalized = normalize_frame(frame)

    assert normalized.min() == pytest.approx(0)
    assert normalized.max() == pytest.approx(1)
    with pytest.raises(ValueError, match="no intensity range"):
        normalize_frame(np.ones((64, 64, 64), np.float32))


def test_selection_grid_is_small_and_correction_only() -> None:
    configs = selection_configurations()

    assert len(configs) == 18
    assert {row.max_pair_distance_um for row in configs} == {12.0}
    assert min(row.min_appearance_gain for row in configs) > 0
    assert max(row.base_lock_probability for row in configs) < 1


def test_selection_key_prioritizes_worst_embryo_delta() -> None:
    def row(min_delta: float, pooled: float):
        return {
            "selection_min_delta_vs_base": min_delta,
            "selection_summary": {
                "proxy_score": pooled,
                "worst_movie": pooled,
            },
            "total_swaps": 1,
            "config": {"min_appearance_gain": 0.05},
        }

    assert robust_selection_key(row(0.0, 0.9)) > robust_selection_key(row(-0.01, 1.0))
