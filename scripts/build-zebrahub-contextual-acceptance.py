from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.temporal_contrastive import zebrahub_external as external  # noqa: E402
from research.temporal_contrastive.verify_zebrahub_contextual_acceptance import (  # noqa: E402
    ACCEPTANCE_COUNT,
    DATASET_ID,
    RUN_ID,
    SOURCE,
    SOURCE_ROLE,
    verify_acceptance,
)


SOURCE_FRAMES = 791
SOURCE_TRACKS_BYTES = 890_580_527
ACCEPTANCE_BLOCKS = (120, 300, 480, 660)
ACCEPTANCE_BLOCK_WIDTH = 6
SEED_BASE = 41_001


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def interleaved_timepoints(
    block_starts: tuple[int, ...], block_width: int
) -> list[int]:
    if not block_starts or block_width <= 0:
        raise ValueError("timepoint blocks must be non-empty and positive")
    result = [
        int(start + offset)
        for offset in range(block_width)
        for start in block_starts
    ]
    if len(set(result)) != len(result) or min(result) < 2 or max(result) > SOURCE_FRAMES - 2:
        raise ValueError("timepoint blocks overlap or escape temporal context")
    return result


@contextlib.contextmanager
def acceptance_source_scope() -> Iterator[None]:
    """Expose the frozen acceptance source only inside this standalone builder."""

    if SOURCE in external.SOURCE_SPECS:
        raise RuntimeError("acceptance source unexpectedly entered the training split")
    external.SOURCE_SPECS[SOURCE] = {
        "role": SOURCE_ROLE,
        "frames": SOURCE_FRAMES,
        "tracks_bytes": SOURCE_TRACKS_BYTES,
    }
    try:
        yield
    finally:
        observed = external.SOURCE_SPECS.pop(SOURCE, None)
        if observed is None:
            raise RuntimeError("acceptance source scope was mutated")


def verified_existing_shard(
    path: Path, *, csv_timepoint: int
) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"partial acceptance shard exists: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_hash = external.sha256_file(path)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("source") == SOURCE
        and manifest.get("source_role") == SOURCE_ROLE
        and manifest.get("csv_timepoint") == int(csv_timepoint)
        and manifest.get("organizer_declared_test_overlap") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_competition_predictions_read") is False
        and manifest.get("leaderboard_used") is False
        and manifest.get("submission_created") is False
        and manifest.get("candidate_context_width") == 18
        and manifest.get("sampling_policy") == external.SAMPLING_POLICY
        and manifest.get("division_quota_fraction") == external.DIVISION_QUOTA_FRACTION
        and manifest.get("maximum_division_source_fraction")
        == external.MAXIMUM_DIVISION_SOURCE_FRACTION
        and manifest.get("shard", {}).get("path") == path.name
        and manifest.get("shard", {}).get("bytes") == path.stat().st_size
        and manifest.get("shard", {}).get("sha256") == shard_hash
    ):
        raise ValueError(f"existing acceptance shard failed verification: {path.name}")
    return manifest


def select_supported_timepoints(
    rows: dict[int, dict[str, Any]], candidates: list[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    rejected: list[dict[str, Any]] = []
    for timepoint in candidates:
        source = rows[timepoint]
        target = rows[timepoint + 1]
        try:
            external.select_dense_transition(
                source["id"],
                source["coords"],
                target["id"],
                target["coords"],
                target["parent_id"],
                max_sources=64,
                max_targets=96,
                seed=SEED_BASE + timepoint,
            )
        except ValueError as error:
            rejected.append({"csv_timepoint": timepoint, "reason": str(error)})
            continue
        selected.append(timepoint)
        if len(selected) == ACCEPTANCE_COUNT:
            break
    if len(selected) != ACCEPTANCE_COUNT:
        raise RuntimeError(
            f"only {len(selected)}/{ACCEPTANCE_COUNT} acceptance transitions are valid"
        )
    return selected, rejected


def filtered_acceptance_csv_frames(
    path: Path, frame_times: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    """Convert ZSNS001 track-level lineage rows to adjacent-frame node links."""

    requested = {int(value) for value in frame_times}
    if not requested or min(requested) < 1:
        raise ValueError("CSV frame times must be positive")
    raw: dict[int, dict[str, list[Any]]] = {
        timepoint: {"track_id": [], "parent_track_id": [], "coords": []}
        for timepoint in requested
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"track_id", "t", "z", "y", "x", "parent_track_id"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError("ZSNS001 track-level CSV schema changed")
        for record in reader:
            timepoint = int(record["t"])
            if timepoint not in requested:
                continue
            raw[timepoint]["track_id"].append(int(record["track_id"]))
            raw[timepoint]["parent_track_id"].append(
                int(record["parent_track_id"])
            )
            raw[timepoint]["coords"].append(
                [float(record["z"]), float(record["y"]), float(record["x"])]
            )
    result: dict[int, dict[str, np.ndarray]] = {}
    for timepoint in sorted(requested):
        track_ids = np.asarray(raw[timepoint]["track_id"], dtype=np.int64)
        parent_tracks = np.asarray(
            raw[timepoint]["parent_track_id"], dtype=np.int64
        )
        coords = np.asarray(raw[timepoint]["coords"], dtype=np.float32).reshape(-1, 3)
        if not len(track_ids):
            raise ValueError(f"ZSNS001 frame {timepoint} contains no track rows")
        if len(np.unique(track_ids)) != len(track_ids):
            raise ValueError(f"ZSNS001 frame {timepoint} repeats a track identifier")
        previous = result.get(timepoint - 1)
        if previous is None:
            parents = parent_tracks
        else:
            previous_ids = previous["id"]
            continuing = np.isin(track_ids, previous_ids)
            parents = np.where(continuing, track_ids, parent_tracks).astype(np.int64)
        result[timepoint] = {
            "id": track_ids,
            "parent_id": parents,
            "coords": coords,
        }
    return result


def build_dataset(cache_root: Path, output_root: Path) -> dict[str, Any]:
    candidates = interleaved_timepoints(ACCEPTANCE_BLOCKS, ACCEPTANCE_BLOCK_WIDTH)
    tracks_url = f"{external.PUBLIC_ROOT}/{SOURCE}_tracks.csv"
    tracks_path = cache_root / SOURCE / f"{SOURCE}_tracks.csv"
    tracks_record = external.download_public_file(
        tracks_url, tracks_path, expected_bytes=SOURCE_TRACKS_BYTES
    )
    requested_frames = sorted(
        {timepoint for candidate in candidates for timepoint in (candidate, candidate + 1)}
    )
    rows = filtered_acceptance_csv_frames(tracks_path, requested_frames)
    selected, rejected = select_supported_timepoints(rows, candidates)
    reader = external.PublicZarrV2Array(
        f"{external.PUBLIC_ROOT}/{SOURCE}.ome.zarr/1",
        cache_root / SOURCE / "level1",
    )
    split_root = output_root / "acceptance"
    split_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    with acceptance_source_scope():
        for index, timepoint in enumerate(selected, start=1):
            shard_path = split_root / f"{SOURCE}-t{timepoint:04d}.npz"
            manifest = verified_existing_shard(
                shard_path, csv_timepoint=timepoint
            )
            if manifest is None:
                manifest = external.build_temporal_patch_shard(
                    SOURCE,
                    csv_timepoint=timepoint,
                    cache_root=cache_root,
                    output_path=shard_path,
                    seed=SEED_BASE + timepoint,
                    preloaded_rows=rows,
                    tracks_record=tracks_record,
                    zarr_reader=reader,
                )
            manifests.append(manifest)
            print(
                json.dumps(
                    {
                        "source": SOURCE,
                        "split": "acceptance",
                        "source_role": SOURCE_ROLE,
                        "completed": index,
                        "required": ACCEPTANCE_COUNT,
                        "csv_timepoint": timepoint,
                        "shard_sha256": manifest["shard"]["sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    records = [
        {
            "path": f"acceptance/{manifest['shard']['path']}",
            "bytes": manifest["shard"]["bytes"],
            "sha256": manifest["shard"]["sha256"],
            "manifest_path": (
                f"acceptance/{Path(manifest['shard']['path']).stem}.manifest.json"
            ),
            "manifest_sha256": external.sha256_file(
                split_root
                / Path(manifest["shard"]["path"]).with_suffix(".manifest.json")
            ),
            "csv_timepoint": manifest["csv_timepoint"],
        }
        for manifest in manifests
    ]
    inventory_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    acceptance = {
        "source": SOURCE,
        "split": "acceptance",
        "source_role": SOURCE_ROLE,
        "required_count": ACCEPTANCE_COUNT,
        "candidate_timepoints": candidates,
        "selected_timepoints": selected,
        "rejected_candidates": rejected,
        "tracks": tracks_record,
        "inventory_sha256": inventory_hash,
        "records": records,
    }
    write_json(output_root / "acceptance-inventory.json", acceptance)

    generator_target = output_root / Path(__file__).name
    source_root = ROOT / "research" / "temporal_contrastive"
    source_paths = (
        source_root / "zebrahub_external.py",
        source_root / "transition_context.py",
        source_root / "patch_model.py",
        source_root / "verify_zebrahub_contextual_dataset.py",
        source_root / "verify_zebrahub_contextual_acceptance.py",
    )
    shutil.copy2(Path(__file__), generator_target)
    for source_path in source_paths:
        shutil.copy2(source_path, output_root / source_path.name)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "selection_policy": "fixed_before_model_weights; one-shot post-selection audit only",
        "organizer_authorization": external.ORGANIZER_AUTHORIZATION,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "model_predictions_read": False,
        "submission_created": False,
        "raw_movie_files_included": False,
        "sampling_policy": external.SAMPLING_POLICY,
        "division_quota_fraction": external.DIVISION_QUOTA_FRACTION,
        "maximum_division_source_fraction": external.MAXIMUM_DIVISION_SOURCE_FRACTION,
        "acceptance": acceptance,
        "generator": {
            "path": generator_target.name,
            "sha256": external.sha256_file(generator_target),
        },
        "sources": {
            source_path.name: {
                "path": source_path.name,
                "sha256": external.sha256_file(output_root / source_path.name),
            }
            for source_path in source_paths
        },
    }
    write_json(output_root / "DATASET_MANIFEST.json", manifest)
    write_json(
        output_root / "dataset-metadata.json",
        {
            "title": "Biohub ZebraHub Contextual Acceptance v1",
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    return verify_acceptance(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / ".biohub" / "cache" / "external" / "zebrahub",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / ".biohub"
            / "staging"
            / "biohub-zebrahub-contextual-acceptance-v1"
        ),
    )
    args = parser.parse_args()
    evidence = build_dataset(args.cache_root, args.output_root)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
