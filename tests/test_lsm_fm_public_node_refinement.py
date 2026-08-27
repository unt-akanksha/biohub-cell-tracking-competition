from __future__ import annotations

import numpy as np

from research.lsm_fm_detection.evaluate_public_node_refinement import (
    CONTROL_NAME,
    PublicNodeStrategy,
    refine_public_points,
    resolve_predeclared_strategy,
    select_public_node_strategy,
)


def gaussian(shape: tuple[int, int, int], center: np.ndarray) -> np.ndarray:
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in shape], indexing="ij"
    )
    squared = sum((axis - center[index]) ** 2 for index, axis in enumerate(axes))
    return np.exp(-0.5 * squared).astype(np.float32)


def summary(matched: int, recalls: tuple[float, float], distance: float = 1.0):
    return {
        "matched_gt_nodes": matched,
        "worst_movie_recall": min(recalls),
        "mean_movie_match_distance_um": distance,
        "rows": [
            {"stem": "44b6_movie", "candidate": {"annotated_node_recall": recalls[0]}},
            {"stem": "6bba_movie", "candidate": {"annotated_node_recall": recalls[1]}},
        ],
    }


def test_control_preserves_public_points_exactly() -> None:
    probability = np.zeros((7, 7, 7), dtype=np.float32)
    points = np.asarray([[2.1, 3.2, 4.3]], dtype=np.float32)
    strategy = PublicNodeStrategy(CONTROL_NAME, 0, 2.0, 0.0)
    refined = refine_public_points(probability, points, strategy)
    np.testing.assert_array_equal(refined, points)


def test_blend_moves_toward_independent_probability_center() -> None:
    center = np.asarray([3.3, 3.2, 3.4], dtype=np.float32)
    probability = gaussian((8, 8, 8), center)
    points = np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32)
    full = refine_public_points(
        probability, points, PublicNodeStrategy("full", 2, 2.0, 1.0)
    )
    half = refine_public_points(
        probability, points, PublicNodeStrategy("half", 2, 2.0, 0.5)
    )
    assert np.linalg.norm(full[0] - center) < np.linalg.norm(points[0] - center)
    np.testing.assert_allclose(half, (points + full) / 2.0, atol=1e-6)


def test_selection_requires_strict_gain_and_bounded_movie_regression() -> None:
    summaries = {
        CONTROL_NAME: summary(100, (0.80, 0.90)),
        "tie": summary(100, (0.81, 0.89), distance=0.5),
        "gain_but_regress": summary(101, (0.79, 0.93)),
        "safe_gain": summary(102, (0.799, 0.922)),
    }
    selected, diagnostics = select_public_node_strategy(summaries)
    assert selected == "safe_gain"
    assert diagnostics["tie"]["selection_passed"] is False
    assert diagnostics["gain_but_regress"]["selection_passed"] is False
    assert diagnostics["safe_gain"]["selection_passed"] is True


def test_selection_fails_closed_without_strict_gain() -> None:
    selected, _ = select_public_node_strategy(
        {
            CONTROL_NAME: summary(100, (0.80, 0.90)),
            "lower_distance_only": summary(100, (0.80, 0.90), distance=0.5),
        }
    )
    assert selected is None


def test_strategy_rejects_invalid_blend() -> None:
    import pytest

    with pytest.raises(ValueError, match="blend"):
        PublicNodeStrategy("invalid", 1, 2.0, 1.01)


def test_predeclared_strategy_is_frozen_and_non_control() -> None:
    strategy = resolve_predeclared_strategy("public_lsm_r2_p2_b025")
    assert strategy == PublicNodeStrategy(
        "public_lsm_r2_p2_b025", radius=2, probability_power=2.0, blend=0.25
    )


def test_predeclared_strategy_fails_closed_for_control_or_unknown() -> None:
    import pytest

    for name in (CONTROL_NAME, "unknown"):
        with pytest.raises(ValueError, match="invalid predeclared"):
            resolve_predeclared_strategy(name)
