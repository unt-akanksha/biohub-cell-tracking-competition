from __future__ import annotations

import json

import numpy as np
import pytest

from research.lsm_fm_detection.evaluate_center_enhancement import (
    CONTROL_CANDIDATE,
    scaled_refinements,
    validate_enhancement_training,
)


def test_scaled_refinements_preserve_control_and_peak_count() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    offsets = np.asarray([[0.4, -0.2, 0.8], [-0.6, 0.1, 0.2]], dtype=np.float32)
    results = scaled_refinements(points, offsets, volume_shape=(8, 8, 8))
    np.testing.assert_array_equal(results[CONTROL_CANDIDATE], points)
    assert all(values.shape == points.shape for values in results.values())
    np.testing.assert_allclose(results["center_enhancement_half"], points + 0.5 * offsets)
    np.testing.assert_allclose(results["center_enhancement_full"], points + offsets)


def test_scaled_refinements_reject_nonzero_control() -> None:
    with pytest.raises(ValueError, match="control scale"):
        scaled_refinements(
            np.ones((1, 3)),
            np.ones((1, 3)),
            volume_shape=(8, 8, 8),
            candidate_scales={CONTROL_CANDIDATE: 0.1},
        )


def test_training_validation_rejects_sparse_unknown_negative_use(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    import hashlib

    payload = {
        "status": "completed",
        "validation_overlap": [],
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "public_predictions_copied": False,
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
        "targets": {"unmatched_peaks_used_as_negatives": True},
    }
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown peaks"):
        validate_enhancement_training(checkpoint, result)
