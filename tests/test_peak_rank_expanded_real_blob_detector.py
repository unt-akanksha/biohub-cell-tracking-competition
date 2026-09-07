from __future__ import annotations

from research.peak_rank_detection import train_expanded_real_blob_detector as variant
from research.peak_rank_detection.model_blob import MODEL_FAMILY


def test_terminal_tagging_is_limited_to_final_terminal(tmp_path) -> None:
    seen = {}

    def writer(path, payload):
        seen[path.name] = payload

    variant.tagged_atomic_json(
        tmp_path / "selection_history.json", {"rows": []}, writer=writer
    )
    variant.tagged_atomic_json(
        tmp_path / "terminal.json", {"status": "accepted_at_audit"}, writer=writer
    )
    assert "model_family" not in seen["selection_history.json"]
    assert seen["terminal.json"]["model_family"] == MODEL_FAMILY


def test_blob_training_lane_is_clean_and_distinct() -> None:
    assert variant.RUN_ID.endswith("blob-peak-rank-v11")
    assert variant.local_shape.RUN_ID.endswith("local-shape-peak-rank-v9")
    assert MODEL_FAMILY == "blob_aware_temporal_peak_rank_v11"
