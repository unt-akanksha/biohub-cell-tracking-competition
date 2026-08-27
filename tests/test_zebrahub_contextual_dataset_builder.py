from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-dataset.py"
SPEC = importlib.util.spec_from_file_location("zebrahub_dataset_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_timepoint_blocks_are_interleaved_across_development() -> None:
    values = builder.interleaved_timepoints((10, 110, 210, 310), 3)

    assert values == [10, 110, 210, 310, 11, 111, 211, 311, 12, 112, 212, 312]
    assert builder.interleaved_timepoints(builder.TRAIN_BLOCKS, 20)[:8] == [
        96,
        216,
        336,
        456,
        97,
        217,
        337,
        457,
    ]
    with pytest.raises(ValueError, match="overlap"):
        builder.interleaved_timepoints((10, 11), 2)


def test_existing_shard_resume_is_hash_and_provenance_bound(tmp_path: Path) -> None:
    path = tmp_path / "ZSNS004-t0100.npz"
    path.write_bytes(b"derived-shard")
    shard_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": "ZSNS004",
        "source_role": "external_pretraining",
        "csv_timepoint": 100,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
        "candidate_context_width": 18,
        "shard": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": shard_hash,
        },
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    accepted = builder.verified_existing_shard(
        path,
        source="ZSNS004",
        role="external_pretraining",
        csv_timepoint=100,
    )

    assert accepted is not None
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="failed verification"):
        builder.verified_existing_shard(
            path,
            source="ZSNS004",
            role="external_pretraining",
            csv_timepoint=100,
        )


def test_builder_source_has_frozen_split_and_no_submission_command() -> None:
    source = BUILDER.read_text(encoding="utf-8")

    assert 'args.train_count != 64 or args.validation_count != 16' in source
    assert 'source="ZSNS004"' in source
    assert 'source="ZSNS005"' in source
    assert '"raw_movie_files_included": False' in source
    assert "kaggle competitions submit" not in source
