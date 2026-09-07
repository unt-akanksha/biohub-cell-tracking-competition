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
    return Namespace(
        output_dir=root / "validation",
        checkpoint=checkpoint,
        training_terminal=terminal,
        baseline_predictions=root / "baseline",
        devices="0,1",
        competition_dir=root / "competition",
        batch_size=1,
        calibration_frames=12,
        max_wall_seconds=100.0,
        disable_d4_tta=False,
    )


def test_training_checkpoint_must_match_terminal(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    assert evaluation.validate_training(checkpoint, terminal)["audit_passed"] is True
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        evaluation.validate_training(checkpoint, terminal)


def test_selection_rejection_keeps_acceptance_closed(tmp_path: Path, monkeypatch) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)
    calls = []

    def fake_launch(_args, stems, *, phase, use_baseline):
        calls.append((phase, use_baseline))
        return [_row(stem, 0.60) for stem in stems]

    monkeypatch.setattr(evaluation, "launch_workers", fake_launch)
    evaluation.orchestrate(args)
    result = json.loads((args.output_dir / "peak_rank_validation.json").read_text())
    assert calls == [("selection", False)]
    assert result["selection_passed"] is False
    assert result["acceptance_opened"] is False
    assert result["promotion_passed"] is False


def test_two_phase_validation_promotes_only_clean_gain(tmp_path: Path, monkeypatch) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)

    def fake_launch(_args, stems, *, phase, use_baseline):
        if phase == "selection":
            return [_row(stem, 0.90) for stem in stems]
        assert use_baseline is True
        return [_row(stem, 0.98, baseline=0.97) for stem in stems]

    monkeypatch.setattr(evaluation, "launch_workers", fake_launch)
    evaluation.orchestrate(args)
    result = json.loads((args.output_dir / "peak_rank_validation.json").read_text())
    assert result["selection_passed"] is True
    assert result["acceptance_opened"] is True
    assert result["promotion_passed"] is True
    assert result["competition_submission_performed"] is False


def test_worker_launcher_requires_exactly_two_devices(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)
    args.devices = "0"
    with pytest.raises(ValueError, match="exactly two"):
        evaluation.launch_workers(
            args, evaluation.SCREEN_STEMS, phase="selection", use_baseline=False
        )
