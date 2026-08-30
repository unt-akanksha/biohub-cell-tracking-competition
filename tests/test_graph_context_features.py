from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from research.temporal_contrastive.graph_context_features import (
    context_tokens,
    physical_nodes,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build-graph-context-relational-archive.py"
SPEC = importlib.util.spec_from_file_location("graph_context_archive_reference", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_runtime_features_exactly_match_sealed_archive_builder() -> None:
    voxel_nodes = {
        10: {"t": 3, "z": 4, "y": 20, "x": 30},
        11: {"t": 4, "z": 5, "y": 16, "x": 30},
        12: {"t": 4, "z": 5, "y": 24, "x": 30},
        20: {"t": 1, "z": 4, "y": 18, "x": 31},
        21: {"t": 2, "z": 4, "y": 19, "x": 30},
        22: {"t": 3, "z": 4, "y": 25, "x": 28},
        23: {"t": 4, "z": 5, "y": 28, "x": 32},
        24: {"t": 5, "z": 6, "y": 30, "x": 33},
    }
    physical = physical_nodes(voxel_nodes)

    expected, expected_mask = BUILDER.context_tokens(
        physical, parent_id=10, existing_child_id=11, proposed_child_id=12
    )
    actual, actual_mask = context_tokens(
        physical, parent_id=10, existing_child_id=11, proposed_child_id=12
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    swapped, swapped_mask = context_tokens(
        physical, parent_id=10, existing_child_id=12, proposed_child_id=11
    )
    np.testing.assert_array_equal(actual, swapped)
    np.testing.assert_array_equal(actual_mask, swapped_mask)
