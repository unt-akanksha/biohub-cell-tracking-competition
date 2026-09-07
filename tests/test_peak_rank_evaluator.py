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
        tta_modes="none,zflip2,rot4,d4",
    )


def test_training_checkpoint_must_match_terminal(tmp_path: Path) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    assert evaluation.validate_training(checkpoint, terminal)["audit_passed"] is True
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        evaluation.validate_training(checkpoint, terminal)


def test_worker_imports_density_calibration_implementation() -> None:
    assert callable(evaluation.density_threshold)


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

    def fake_launch(_args, stems, *, phase, use_baseline, tta_mode):
        calls.append((phase, use_baseline, tta_mode))
        return [_row(stem, 0.60) for stem in stems]

    monkeypatch.setattr(evaluation, "launch_workers", fake_launch)
    evaluation.orchestrate(args)
    result = json.loads((args.output_dir / "peak_rank_validation.json").read_text())
    assert calls == [
        ("tta-calibration-none", False, "none"),
        ("tta-calibration-zflip2", False, "zflip2"),
        ("tta-calibration-rot4", False, "rot4"),
        ("tta-calibration-d4", False, "d4"),
    ]
    assert result["selection_passed"] is False
    assert result["acceptance_opened"] is False
    assert result["promotion_passed"] is False


def test_two_phase_validation_promotes_only_clean_gain(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint, terminal = _training_files(tmp_path)
    args = _args(tmp_path, checkpoint, terminal)

    def fake_launch(_args, stems, *, phase, use_baseline, tta_mode):
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
        )


def test_tta_selection_prefers_cheapest_nonregressing_mode() -> None:
    summaries = {
        "none": {"annotated_node_recall": 0.966, "worst_movie_recall": 0.91},
        "zflip2": {"annotated_node_recall": 0.967, "worst_movie_recall": 0.911},
        "rot4": {"annotated_node_recall": 0.968, "worst_movie_recall": 0.912},
        "d4": {"annotated_node_recall": 0.969, "worst_movie_recall": 0.914},
    }
    selected, gate = evaluation.select_tta_mode(summaries)
    assert selected == "none"
    assert gate["eligible_modes"] == ["none", "zflip2", "rot4", "d4"]


def test_tta_selection_rejects_fast_mode_with_recall_loss() -> None:
    summaries = {
        "none": {"annotated_node_recall": 0.95, "worst_movie_recall": 0.89},
        "zflip2": {"annotated_node_recall": 0.951, "worst_movie_recall": 0.891},
        "rot4": {"annotated_node_recall": 0.968, "worst_movie_recall": 0.912},
        "d4": {"annotated_node_recall": 0.969, "worst_movie_recall": 0.914},
    }
    selected, _gate = evaluation.select_tta_mode(summaries)
    assert selected == "rot4"
