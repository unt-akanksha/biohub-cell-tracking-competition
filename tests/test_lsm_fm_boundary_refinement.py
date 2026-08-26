from __future__ import annotations

from research.lsm_fm_detection.evaluate_boundary_refinement import (
    BOUNDARY_STRATEGIES,
    CONTROL_STRATEGY,
)


def test_boundary_sweep_is_small_global_and_anchored_to_near_gate_control() -> None:
    assert BOUNDARY_STRATEGIES[0].name == CONTROL_STRATEGY
    assert (BOUNDARY_STRATEGIES[0].radius, BOUNDARY_STRATEGIES[0].probability_power) == (
        2,
        2.0,
    )
    assert len(BOUNDARY_STRATEGIES) == 6
    assert len({strategy.name for strategy in BOUNDARY_STRATEGIES}) == len(
        BOUNDARY_STRATEGIES
    )
    assert all(strategy.method == "weighted" for strategy in BOUNDARY_STRATEGIES)
    assert all(strategy.intensity_power == 0.0 for strategy in BOUNDARY_STRATEGIES)


def test_boundary_sweep_only_expands_support_or_flattens_probability_weight() -> None:
    control = BOUNDARY_STRATEGIES[0]
    for candidate in BOUNDARY_STRATEGIES[1:]:
        assert candidate.radius >= control.radius
        assert candidate.probability_power <= control.probability_power
