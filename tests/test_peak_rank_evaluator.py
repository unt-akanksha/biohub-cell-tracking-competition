import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from research.peak_rank_detection import evaluate_peak_rank_detector as evaluation


def _training_files(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "peak_rank_detector.pt"
    checkpoint.write_bytes(b"weights")
    terminal = root / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "status": "accepted_at_audit",
                "selection_passed": True,
                "audit_opened": True,
                "audit_passed": True,
                "parameter_count": 38_381_478,
                "widths": [96, 192, 384, 768],
                "depths": [3, 3, 9, 3],
                "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
                "competition_test_data_read": False,
                "public_predictions_read": False,
                "public_notebook_weights_read": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "authorized_for_submission": False,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, terminal


def _row(stem: str, recall: float, *, baseline: float | None = None) -> dict:
    annotated = 100
    result = {
        "stem": stem,
        "candidate": {
            "annotated_gt_nodes": annotated,
            "matched_gt_nodes": round(annotated * recall),
            "annotated_node_recall": recall,
        },
    }
    if baseline is not None:
        result["baseline_raw_graph"] = {"annotated_node_recall": baseline}
        result["annotated_recall_delta"] = recall - baseline
    return result


def _args(root: Path, checkpoint: Path, terminal: Path) -> Namespace:
    calibration = root / "threshold_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "run_id": "synthetic-complete-global-peak-threshold-v1",
                "status": "calibrated",
                "threshold_policy": "synthetic_selection_micro_detection_jaccard",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "training_terminal_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
                "parameter_count": 38_381_478,
                "complete_synthetic_labels_read": True,
                "competition_train_data_read": False,
                "competition_test_data_read": False,
                "organizer_estimated_node_count_read": False,
                "organizer_estimated_node_count_used_for_threshold": False,
                "public_predictions_read": False,
                "public_notebook_weights_read": False,
                "public_leaderboard_used_for_selection": False,
                "thresholds": {
                    mode: {
                        "threshold": 0.5,
                        "tta_mode": mode,
                        "tta_views": views,
                        "complete_synthetic_examples": 24,
                        "total_truth_nodes": 100,
                        "selected_predictions": 100,
                        "precision": 0.9,
                        "recall": 0.9,
                        "detection_jaccard": 0.82,
                    }
                    for mode, views in {
                        "none": 1,
                        "zflip2": 2,
                        "rot4": 4,
                        "d4": 8,
                    }.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return Namespace(
        output_dir=root / "validation",
        checkpoint=checkpoint,
        training_terminal=terminal,
        threshold_calibration=calibration,
        baseline_predictions=root / "baseline",
        devices="0,1",
        competition_dir=root / "competition",
        batch_size=1,
        max_wall_seconds=100.0,
        tta_modes="none,zflip2,rot4,d4",
    )


def test_training_checkpoint_must_match_terminal(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    assert evaluation.validate_training(checkpoint, terminal)["audit_passed"] is True
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        evaluation.validate_training(checkpoint, terminal)


def test_threshold_calibration_rejects_organizer_count_use(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)
    payload = json.loads(args.threshold_calibration.read_text(encoding="utf-8"))
    payload["organizer_estimated_node_count_used_for_threshold"] = True
    args.threshold_calibration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="clean contract"):
        evaluation.validate_threshold_calibration(
            args.threshold_calibration,
            checkpoint=checkpoint,
            training_terminal=terminal,
            parameter_count=38_381_478,
        )


def test_training_rejects_an_unknown_model_family(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    payload["model_family"] = "unreviewed_external_detector"
    terminal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted clean"):
        evaluation.validate_training(checkpoint, terminal)


def test_selection_rejection_keeps_acceptance_closed(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)
    calls = []

    def fake_launch(_args, stems, *, phase, use_baseline, tta_mode, peak_threshold):
        assert peak_threshold == 0.5
        calls.append((phase, use_baseline, tta_mode))
        return [_row(stem, 0.60) for stem in stems]

    monkeypatch.setattr(evaluation, "launch_workers", fake_launch)
    evaluation.orchestrate(args)
    result = json.loads((args.output_dir / "peak_rank_validation.json").read_text())
    assert calls == [
        ("selection-none", False, "none"),
    ]
    assert result["selection_passed"] is False
    assert result["acceptance_opened"] is False
    assert result["promotion_passed"] is False


def test_two_phase_validation_promotes_only_clean_gain(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)

    def fake_launch(_args, stems, *, phase, use_baseline, tta_mode, peak_threshold):
        assert peak_threshold == 0.5
        if phase.startswith("tta-calibration") or phase.startswith("selection"):
            return [_row(stem, 0.90) for stem in stems]
        assert use_baseline is True
        return [_row(stem, 0.98, baseline=0.97) for stem in stems]

    monkeypatch.setattr(evaluation, "launch_workers", fake_launch)
    evaluation.orchestrate(args)
    result = json.loads((args.output_dir / "peak_rank_validation.json").read_text())
    assert result["selection_passed"] is True
    assert result["selected_tta_mode"] == "none"
    assert result["selected_tta_views"] == 1
    assert result["acceptance_opened"] is True
    assert result["promotion_passed"] is True
    assert result["competition_submission_performed"] is False


def test_worker_launcher_requires_exactly_two_devices(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)
    args.devices = "0"
    with pytest.raises(ValueError, match="exactly two"):
        evaluation.launch_workers(
            args,
            evaluation.SCREEN_STEMS,
            phase="selection",
            use_baseline=False,
            tta_mode="d4",
            peak_threshold=0.5,
        )


def test_tta_selection_prefers_cheapest_nonregressing_mode() -> None:
    summaries = {
        "none": {"detection_jaccard": 0.966, "recall": 0.91},
        "zflip2": {"detection_jaccard": 0.967, "recall": 0.911},
        "rot4": {"detection_jaccard": 0.968, "recall": 0.912},
        "d4": {"detection_jaccard": 0.969, "recall": 0.914},
    }
    selected, gate = evaluation.select_tta_mode(summaries)
    assert selected == "none"
    assert gate["eligible_modes"] == ["none", "zflip2", "rot4", "d4"]


def test_tta_selection_rejects_fast_mode_with_recall_loss() -> None:
    summaries = {
        "none": {"detection_jaccard": 0.95, "recall": 0.89},
        "zflip2": {"detection_jaccard": 0.951, "recall": 0.891},
        "rot4": {"detection_jaccard": 0.968, "recall": 0.912},
        "d4": {"detection_jaccard": 0.969, "recall": 0.914},
    }
    selected, _gate = evaluation.select_tta_mode(summaries)
    assert selected == "rot4"
