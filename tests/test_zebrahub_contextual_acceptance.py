from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

from research.temporal_contrastive.verify_zebrahub_contextual_acceptance import (
    RUN_ID,
    SOURCE,
    SOURCE_ROLE,
    verify_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-acceptance.py"
SPEC = importlib.util.spec_from_file_location("zebrahub_acceptance_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_acceptance_windows_are_fixed_across_development() -> None:
    values = builder.interleaved_timepoints(
        builder.ACCEPTANCE_BLOCKS, builder.ACCEPTANCE_BLOCK_WIDTH
    )

    assert values[:8] == [120, 300, 480, 660, 121, 301, 481, 661]
    assert len(values) == 24
    assert len(set(values)) == 24
    with pytest.raises(ValueError, match="overlap"):
        builder.interleaved_timepoints((10, 11), 2)


def test_acceptance_source_never_enters_frozen_training_specs() -> None:
    assert SOURCE not in builder.external.SOURCE_SPECS
    with builder.acceptance_source_scope():
        assert builder.external.SOURCE_SPECS[SOURCE] == {
            "role": SOURCE_ROLE,
            "frames": 791,
            "tracks_bytes": 890_580_527,
        }
    assert SOURCE not in builder.external.SOURCE_SPECS


def test_acceptance_resume_is_hash_and_provenance_bound(tmp_path: Path) -> None:
    path = tmp_path / "ZSNS001-t0120.npz"
    path.write_bytes(b"derived-acceptance-shard")
    shard_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "csv_timepoint": 120,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
        "candidate_context_width": 18,
        "sampling_policy": builder.external.SAMPLING_POLICY,
        "division_quota_fraction": builder.external.DIVISION_QUOTA_FRACTION,
        "maximum_division_source_fraction": (
            builder.external.MAXIMUM_DIVISION_SOURCE_FRACTION
        ),
        "shard": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": shard_hash,
        },
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert builder.verified_existing_shard(path, csv_timepoint=120) is not None
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="failed verification"):
        builder.verified_existing_shard(path, csv_timepoint=120)


def test_track_level_csv_converts_continuations_and_daughters(tmp_path: Path) -> None:
    path = tmp_path / "tracks.csv"
    path.write_text(
        "track_id,t,z,y,x,parent_track_id\n"
        "10,120,1,2,3,-1\n"
        "20,120,4,5,6,-1\n"
        "10,121,2,3,4,-1\n"
        "21,121,5,6,7,20\n",
        encoding="utf-8",
    )

    rows = builder.filtered_acceptance_csv_frames(path, [120, 121])

    assert np.array_equal(rows[120]["id"], np.asarray([10, 20]))
    assert np.array_equal(rows[121]["id"], np.asarray([10, 21]))
    assert np.array_equal(rows[121]["parent_id"], np.asarray([10, 20]))
    assert rows[121]["coords"].shape == (2, 3)


def test_acceptance_verifier_rejects_opened_model_evidence(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "selection_policy": "fixed_before_model_weights; one-shot post-selection audit only",
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "model_predictions_read": True,
        "submission_created": False,
        "raw_movie_files_included": False,
        "sampling_policy": builder.external.SAMPLING_POLICY,
        "division_quota_fraction": builder.external.DIVISION_QUOTA_FRACTION,
        "maximum_division_source_fraction": (
            builder.external.MAXIMUM_DIVISION_SOURCE_FRACTION
        ),
    }
    (tmp_path / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="global provenance"):
        verify_acceptance(tmp_path)


def test_acceptance_builder_has_no_submission_or_training_path() -> None:
    source = BUILDER.read_text(encoding="utf-8")

    assert 'SOURCE = "ZSNS001"' not in source  # imported from the verifier contract
    assert "model_predictions_read\": False" in source
    assert "one-shot post-selection audit only" in source
    assert "kaggle competitions submit" not in source.casefold()
