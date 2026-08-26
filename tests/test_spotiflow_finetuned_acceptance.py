from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.spotiflow_biohub.evaluate_finetuned_detector import (
    validate_training_result,
)


def test_training_result_binds_changed_best_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    best = model_dir / "best.pt"
    best.write_bytes(b"learned checkpoint")
    import hashlib

    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "weights_changed": True,
                "best_weight_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert validate_training_result(model_dir, result)["weights_changed"] is True


def test_training_result_rejects_unbound_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "best.pt").write_bytes(b"other")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "weights_changed": True,
                "best_weight_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_training_result(model_dir, result)
