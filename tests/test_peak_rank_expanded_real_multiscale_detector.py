from pathlib import Path

from research.peak_rank_detection import (
    train_expanded_real_multiscale_detector as variant,
)
from research.peak_rank_detection.model_multiscale import MODEL_FAMILY


def test_multiscale_trainer_contract() -> None:
    assert variant.RUN_ID.endswith("multiscale-blob-global-peak-rank-v15")
    assert MODEL_FAMILY == "multiscale_blob_global_context_temporal_peak_rank_v15"


def test_terminal_tag_records_training_only_scale_selection() -> None:
    written = {}

    def writer(path: Path, payload: dict) -> None:
        written[path.name] = payload

    variant.tagged_atomic_json(
        Path("selection_history.json"), {"status": "running"}, writer=writer
    )
    variant.tagged_atomic_json(
        Path("terminal.json"), {"status": "complete"}, writer=writer
    )
    assert "model_family" not in written["selection_history.json"]
    terminal = written["terminal.json"]
    assert terminal["model_family"] == MODEL_FAMILY
    assert terminal["blob_bands"] == [[3, 9], [5, 13], [7, 15]]
    assert terminal["blob_scale_selection_role"] == "expanded_real_optimization_only"
