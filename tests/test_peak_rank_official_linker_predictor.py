from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from research.peak_rank_detection.predict_with_official_linker import (
    parse_slice,
    selected_names,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "research"
    / "peak_rank_detection"
    / "predict_with_official_linker.py"
)


def test_production_predictor_is_two_worker_label_free_and_non_submitting() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "predict_movie_detection_cache",
        "predict_video_with_external_detections",
        "torch.cuda.device_count() != 1",
        "competition_train_labels_read",
        "competition_test_labels_read",
        "public_predictions_copied",
        "worker_manifest.json",
    ):
        assert required in source
    for forbidden in (
        "kaggle competitions submit",
        "submission.csv",
        "leaderboard score",
    ):
        assert forbidden not in source


def test_worker_partition_is_complete_and_disjoint(tmp_path: Path) -> None:
    names = [f"movie-{index}.zarr" for index in range(7)]
    for name in names:
        (tmp_path / name).mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"test": names}]), encoding="utf-8")
    first = selected_names(tmp_path, splits, 0, 0, 2)
    second = selected_names(tmp_path, splits, 0, 1, 2)
    assert set(first).isdisjoint(second)
    assert sorted(first + second) == sorted(names)


def test_worker_partition_rejects_missing_and_duplicates(tmp_path: Path) -> None:
    (tmp_path / "one.zarr").mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"test": ["one.zarr", "one.zarr"]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        selected_names(tmp_path, splits, 0, 0, 2)
    splits.write_text(json.dumps([{"test": ["missing.zarr"]}]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        selected_names(tmp_path, splits, 0, 0, 2)


def test_public_notebook_slice_contract_is_supported(tmp_path: Path) -> None:
    names = [f"movie-{index}" for index in range(6)]
    for name in names:
        (tmp_path / f"{name}.zarr").mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"test": names}]), encoding="utf-8")
    assert selected_names(
        tmp_path, splits, 0, 0, 1, parse_slice("1::2")
    ) == names[1::2]
