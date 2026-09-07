from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.peak_rank_detection.predict_with_official_linker import (
    EXPECTED_ASSOCIATION_CONFIG,
    load_attributed_association_stack,
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
        "--peak-tta-mode",
        "projected_worker_seconds",
        "RUNTIME_PROJECTION_SAFETY_FACTOR",
        "load_attributed_association_stack",
        "secondary_model",
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


def test_attributed_association_stack_loads_both_frozen_members(
    tmp_path: Path, monkeypatch
) -> None:
    primary = tmp_path / "primary.pt"
    secondary = tmp_path / "secondary.pt"
    primary.write_bytes(b"primary")
    secondary.write_bytes(b"secondary")
    environment = {
        "BIOHUB_SECONDARY_WEIGHTS": str(secondary),
        "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
        "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
        "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
        "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1.0",
        "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
        "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
        "BIOHUB_EDGE_FEATURE_TTA": "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    official = SimpleNamespace(
        load_model=lambda path, _device: (path.name, 5, (1, 4, 4))
    )
    model, window, downsample, kwargs, manifest = load_attributed_association_stack(
        official, primary, "cuda:0"
    )
    assert model == "primary.pt"
    assert window == 5 and downsample == (1, 4, 4)
    assert kwargs["secondary_model"] == "secondary.pt"
    assert kwargs["secondary_link_mode"] == "low_margin_consensus"
    assert manifest["edge_feature_tta"] is True
    assert manifest["bidirectional_edge_weight"] == 0.15


def test_attributed_association_stack_rejects_config_drift(
    tmp_path: Path, monkeypatch
) -> None:
    primary = tmp_path / "primary.pt"
    secondary = tmp_path / "secondary.pt"
    primary.write_bytes(b"primary")
    secondary.write_bytes(b"secondary")
    names = {
        "secondary_edge_weight": "BIOHUB_SECONDARY_EDGE_WEIGHT",
        "secondary_detection_weight": "BIOHUB_SECONDARY_DETECTION_WEIGHT",
        "secondary_mix_temperature": "BIOHUB_SECONDARY_MIX_TEMPERATURE",
        "secondary_low_margin_max": "BIOHUB_SECONDARY_LOW_MARGIN_MAX",
        "edge_candidate_threshold": "BIOHUB_DUAL_SEED_EDGE_THRESHOLD",
        "bidirectional_edge_weight": "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT",
    }
    monkeypatch.setenv("BIOHUB_SECONDARY_WEIGHTS", str(secondary))
    monkeypatch.setenv("BIOHUB_SECONDARY_LINK_MODE", "low_margin_consensus")
    monkeypatch.setenv("BIOHUB_EDGE_FEATURE_TTA", "1")
    for name, environment_name in names.items():
        monkeypatch.setenv(environment_name, str(EXPECTED_ASSOCIATION_CONFIG[name]))
    monkeypatch.setenv("BIOHUB_SECONDARY_EDGE_WEIGHT", "0.25")
    official = SimpleNamespace(load_model=lambda _path, _device: (object(), 5, (1, 4, 4)))
    with pytest.raises(ValueError, match="config drift"):
        load_attributed_association_stack(official, primary, "cuda:0")
