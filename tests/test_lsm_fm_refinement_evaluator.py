from __future__ import annotations

from research.lsm_fm_detection.evaluate_localization_refinement import (
    CONTROL_STRATEGY,
    select_global_strategy,
)


def summary(pooled: float, rows: dict[str, float], distance: float = 2.0) -> dict:
    return {
        "annotated_node_recall": pooled,
        "worst_movie_recall": min(rows.values()),
        "mean_movie_match_distance_um": distance,
        "rows": [
            {"stem": stem, "candidate": {"annotated_node_recall": recall}}
            for stem, recall in rows.items()
        ],
    }


def test_selects_global_worst_movie_improvement_without_other_regression() -> None:
    summaries = {
        CONTROL_STRATEGY: summary(0.88, {"a": 0.90, "hard": 0.62}),
        "joint": summary(0.89, {"a": 0.895, "hard": 0.68}, distance=1.8),
        "overfit": summary(0.91, {"a": 0.87, "hard": 0.72}, distance=1.5),
    }
    selected, diagnostics = select_global_strategy(summaries)
    assert selected == "joint"
    assert diagnostics["joint"]["selection_passed"] is True
    assert diagnostics["overfit"]["selection_passed"] is False


def test_acceptance_stays_sealed_when_no_strategy_crosses_worst_gate() -> None:
    summaries = {
        CONTROL_STRATEGY: summary(0.88, {"a": 0.90, "hard": 0.62}),
        "small_gain": summary(0.885, {"a": 0.90, "hard": 0.64}),
    }
    selected, diagnostics = select_global_strategy(summaries)
    assert selected is None
    assert all(not value["selection_passed"] for value in diagnostics.values())
