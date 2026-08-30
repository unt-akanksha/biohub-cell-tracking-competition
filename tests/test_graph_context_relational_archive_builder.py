from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-context-relational-archive.py"
SPEC = importlib.util.spec_from_file_location("graph_context_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_nodes() -> dict[int, tuple[int, np.ndarray]]:
    rows: dict[int, tuple[int, np.ndarray]] = {
        10: (4, np.asarray((10.0, 10.0, 10.0), dtype=np.float32)),
        11: (5, np.asarray((11.0, 8.0, 10.0), dtype=np.float32)),
        12: (5, np.asarray((11.0, 12.0, 10.0), dtype=np.float32)),
    }
    node_id = 100
    for timepoint in range(2, 7):
        for offset in range(12):
            rows[node_id] = (
                timepoint,
                np.asarray((10.0, float(offset), 10.0), dtype=np.float32),
            )
            node_id += 1
    return rows


def test_context_is_exactly_daughter_order_invariant_and_bounded() -> None:
    nodes = synthetic_nodes()
    first, first_mask = MODULE.context_tokens(
        nodes, parent_id=10, existing_child_id=11, proposed_child_id=12
    )
    swapped, swapped_mask = MODULE.context_tokens(
        nodes, parent_id=10, existing_child_id=12, proposed_child_id=11
    )

    np.testing.assert_array_equal(first, swapped)
    np.testing.assert_array_equal(first_mask, swapped_mask)
    assert first.shape == (MODULE.CONTEXT_TOKEN_COUNT, MODULE.CONTEXT_FEATURE_WIDTH)
    assert first_mask.shape == (MODULE.CONTEXT_TOKEN_COUNT,)
    assert first_mask[:3].all()
    assert int(first_mask.sum()) <= MODULE.CONTEXT_TOKEN_COUNT
    assert np.isfinite(first).all()


def test_context_uses_fixed_nearest_detection_budget_without_edge_features() -> None:
    nodes = synthetic_nodes()
    features, mask = MODULE.context_tokens(
        nodes, parent_id=10, existing_child_id=11, proposed_child_id=12
    )

    assert MODULE.NODE_ARRAY_PATHS == (
        "nodes/ids",
        "nodes/props/t/values",
        "nodes/props/z/values",
        "nodes/props/y/values",
        "nodes/props/x/values",
    )
    assert all("edge" not in path for path in MODULE.NODE_ARRAY_PATHS)
    assert MODULE.CONTEXT_TOKEN_COUNT == 43
    assert features[mask, -1].sum() <= 5 * MODULE.CONTEXT_NEIGHBORS_PER_TIME
    assert features[0, -3] == 1.0
    assert features[1, -2] == features[2, -2] == 1.0


def test_builder_contract_binds_sealed_sources_and_forbids_submission() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert MODULE.SOURCE_ARCHIVE_SHA256 == "66a822bce0c60d06f6a2b60ada313f0d4d55062de1f84fb60bded4ae456266c2"
    assert MODULE.INVENTORY_SHA256 == "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
    assert '"context_edge_arrays_read": []' in source
    assert '"context_labels_used": False' in source
    assert '"audit_labels_scored": False' in source
    assert '"competition_test_data_read": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source
