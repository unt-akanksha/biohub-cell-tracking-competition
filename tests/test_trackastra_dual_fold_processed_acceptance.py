from __future__ import annotations

from research.trackastra_graph.dual_fold_processed_acceptance import (
    FROZEN_ASSOCIATION_CONFIGURATION,
    configuration_sha256,
)


def test_processed_acceptance_configuration_is_frozen_and_conservative() -> None:
    assert FROZEN_ASSOCIATION_CONFIGURATION == {
        "method": "raw_confidence_hybrid",
        "edge_threshold": 0.08,
        "base_lock_probability": 0.98,
        "base_keep_probability": 0.80,
        "base_bonus": 0.05,
        "division_threshold": 0.18,
        "division_ratio": 0.50,
        "base_division_keep_probability": 0.95,
        "base_pseudo_probability": 0.80,
    }
    assert configuration_sha256(FROZEN_ASSOCIATION_CONFIGURATION) == (
        "d1aa8d32f7bfff53cb3e1a53a9afe0d9e029b8b5bf5e431b820cabc90c19e791"
    )
