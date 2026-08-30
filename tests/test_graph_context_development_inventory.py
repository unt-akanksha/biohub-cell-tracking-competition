from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-context-development-inventory.py"
SPEC = importlib.util.spec_from_file_location("graph_context_dev_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_development_inventory_is_node_only_and_count_preserving() -> None:
    source = ROOT / ".biohub/results/competition-relational-division-development-inventory-v1.json"
    prediction_root = (
        ROOT
        / ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema"
        / "tracking_repo/predictions/unknown/unet_transformer_val/split_0"
    )

    result = MODULE.build_inventory(source, prediction_root)
    rows = [row for movie in result["movies"] for row in movie["candidates"]]

    assert result["run_id"] == MODULE.RUN_ID
    assert result["summary"]["rows"] == len(rows) == 225
    assert result["summary"]["inference_geometry_eligible_rows"] == 9
    assert result["summary"]["inference_geometry_eligible_positives"] == 3
    assert result["graph_context_token_count"] == 43
    assert result["graph_context_feature_width"] == 8
    assert result["graph_context_edge_arrays_read"] == []
    assert result["graph_context_edges_read"] is False
    assert result["graph_context_labels_used"] is False
    assert result["authorized_for_graph_context_probe_scoring"] is True
    assert result["authorized_for_submission"] is False
    assert all(len(row["graph_context_features"]) == 43 for row in rows)
    assert all(len(row["graph_context_mask"]) == 43 for row in rows)


def test_builder_never_submits_or_reads_test_data() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"competition_test_data_read") is False' in source
    assert '"public_leaderboard_used_for_selection") is False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source
