from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.temporal_contrastive.zebrahub_external import (  # noqa: E402
    DIVISION_QUOTA_FRACTION,
    MAXIMUM_DIVISION_SOURCE_FRACTION,
    ORGANIZER_AUTHORIZATION,
    PUBLIC_ROOT,
    SAMPLING_POLICY,
    SOURCE_SPECS,
    PublicZarrV2Array,
    build_temporal_patch_shard,
    download_public_file,
    filtered_csv_frames,
    select_dense_transition,
    sha256_file,
)
from research.temporal_contrastive.verify_zebrahub_contextual_dataset import (  # noqa: E402
    verify_dataset,
)


RUN_ID = "zebrahub-contextual-shards-v1"
DATASET_ID = "indarkarhana/biohub-zebrahub-contextual-shards-v1"
TRAIN_BLOCKS = (96, 216, 336, 456)
VALIDATION_BLOCKS = (96, 236, 376, 516)


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
    if len(set(result)) != len(result) or min(result) < 2 or max(result) > 598:
        raise ValueError("timepoint blocks overlap or escape temporal context")
    return result


def verified_existing_shard(
    path: Path,
    *,
    source: str,
    role: str,
    csv_timepoint: int,
) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"partial derived shard exists: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("source") == source
        and manifest.get("source_role") == role
        and manifest.get("csv_timepoint") == int(csv_timepoint)
        and manifest.get("organizer_declared_test_overlap") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_competition_predictions_read") is False
        and manifest.get("leaderboard_used") is False
        and manifest.get("submission_created") is False
        and manifest.get("candidate_context_width") == 18
        and manifest.get("sampling_policy") == SAMPLING_POLICY
        and manifest.get("division_quota_fraction") == DIVISION_QUOTA_FRACTION
        and manifest.get("maximum_division_source_fraction")
        == MAXIMUM_DIVISION_SOURCE_FRACTION
        and manifest.get("shard", {}).get("path") == path.name
        and manifest.get("shard", {}).get("bytes") == path.stat().st_size
        and manifest.get("shard", {}).get("sha256") == actual_hash
    ):
        raise ValueError(f"existing derived shard failed verification: {path.name}")
    return manifest


def select_supported_timepoints(
    rows: dict[int, dict[str, Any]],
    candidates: list[int],
    *,
    required_count: int,
    seed_base: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    rejected: list[dict[str, Any]] = []
    for timepoint in candidates:
        source = rows[timepoint]
        target = rows[timepoint + 1]
        try:
            select_dense_transition(
                source["id"],
                source["coords"],
                target["id"],
                target["coords"],
                target["parent_id"],
                max_sources=64,
                max_targets=96,
                seed=seed_base + timepoint,
            )
        except ValueError as error:
            rejected.append({"csv_timepoint": timepoint, "reason": str(error)})
            continue
        selected.append(timepoint)
        if len(selected) == required_count:
            break
    if len(selected) != required_count:
        raise RuntimeError(
            f"only {len(selected)}/{required_count} candidate transitions are valid"
        )
    return selected, rejected


def build_source(
    *,
    source: str,
    role: str,
    candidates: list[int],
    required_count: int,
    seed_base: int,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    source_role = str(spec["role"])
    tracks_url = f"{PUBLIC_ROOT}/{source}_tracks.csv"
    tracks_path = cache_root / source / f"{source}_tracks.csv"
    tracks_record = download_public_file(
        tracks_url, tracks_path, expected_bytes=int(spec["tracks_bytes"])
    )
    requested_frames = sorted(
        {timepoint for candidate in candidates for timepoint in (candidate, candidate + 1)}
    )
    rows = filtered_csv_frames(tracks_path, requested_frames)
    selected, rejected = select_supported_timepoints(
        rows,
        candidates,
        required_count=required_count,
        seed_base=seed_base,
    )
    reader = PublicZarrV2Array(
        f"{PUBLIC_ROOT}/{source}.ome.zarr/1", cache_root / source / "level1"
    )
    role_root = output_root / role
    role_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for index, timepoint in enumerate(selected, start=1):
        shard_path = role_root / f"{source}-t{timepoint:04d}.npz"
        manifest = verified_existing_shard(
            shard_path,
            source=source,
            role=source_role,
            csv_timepoint=timepoint,
        )
        if manifest is None:
            manifest = build_temporal_patch_shard(
                source,
                csv_timepoint=timepoint,
                cache_root=cache_root,
                output_path=shard_path,
                seed=seed_base + timepoint,
                preloaded_rows=rows,
                tracks_record=tracks_record,
                zarr_reader=reader,
            )
        manifests.append(manifest)
        print(
            json.dumps(
                {
                    "source": source,
                    "split": role,
                    "source_role": source_role,
                    "completed": index,
                    "required": required_count,
                    "csv_timepoint": timepoint,
                    "shard_sha256": manifest["shard"]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    records = [
        {
            "path": f"{role}/{manifest['shard']['path']}",
            "bytes": manifest["shard"]["bytes"],
            "sha256": manifest["shard"]["sha256"],
            "manifest_path": (
                f"{role}/{Path(manifest['shard']['path']).stem}.manifest.json"
            ),
            "manifest_sha256": sha256_file(
                role_root / Path(manifest["shard"]["path"]).with_suffix(
                    ".manifest.json"
                )
            ),
            "csv_timepoint": manifest["csv_timepoint"],
        }
        for manifest in manifests
    ]
    inventory_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "source": source,
        "split": role,
        "source_role": source_role,
        "required_count": required_count,
        "candidate_timepoints": candidates,
        "selected_timepoints": selected,
        "rejected_candidates": rejected,
        "tracks": tracks_record,
        "inventory_sha256": inventory_hash,
        "records": records,
    }
    write_json(output_root / f"{role}-inventory.json", result)
    return result


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
            / "biohub-zebrahub-contextual-shards-v1"
        ),
    )
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    args = parser.parse_args()
    if args.train_count != 64 or args.validation_count != 16:
        raise ValueError("the frozen v1 external split requires exactly 64/16 shards")
    train_candidates = interleaved_timepoints(TRAIN_BLOCKS, 20)
    validation_candidates = interleaved_timepoints(VALIDATION_BLOCKS, 6)
    training = build_source(
        source="ZSNS004",
        role="train",
        candidates=train_candidates,
        required_count=args.train_count,
        seed_base=44_004,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    validation = build_source(
        source="ZSNS005",
        role="validation",
        candidates=validation_candidates,
        required_count=args.validation_count,
        seed_base=45_005,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    if {row["sha256"] for row in training["records"]} & {
        row["sha256"] for row in validation["records"]
    }:
        raise RuntimeError("ZebraHub train and validation derived shards overlap")
    generator_target = args.output_root / Path(__file__).name
    source_root = ROOT / "research" / "temporal_contrastive"
    source_paths = (
        source_root / "zebrahub_external.py",
        source_root / "transition_context.py",
        source_root / "patch_model.py",
        source_root / "verify_zebrahub_contextual_dataset.py",
    )
    shutil.copy2(Path(__file__), generator_target)
    for source_path in source_paths:
        shutil.copy2(source_path, args.output_root / source_path.name)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "organizer_authorization": ORGANIZER_AUTHORIZATION,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
        "raw_movie_files_included": False,
        "sampling_policy": SAMPLING_POLICY,
        "division_quota_fraction": DIVISION_QUOTA_FRACTION,
        "maximum_division_source_fraction": MAXIMUM_DIVISION_SOURCE_FRACTION,
        "training": training,
        "validation": validation,
        "generator": {
            "path": generator_target.name,
            "sha256": sha256_file(generator_target),
        },
        "sources": {
            source_path.name: {
                "path": source_path.name,
                "sha256": sha256_file(args.output_root / source_path.name),
            }
            for source_path in source_paths
        },
    }
    write_json(args.output_root / "DATASET_MANIFEST.json", manifest)
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub ZebraHub Contextual Shards v1",
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    verification = verify_dataset(args.output_root)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "train_shards": len(training["records"]),
                "validation_shards": len(validation["records"]),
                "manifest_sha256": sha256_file(
                    args.output_root / "DATASET_MANIFEST.json"
                ),
                "verification": verification,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
