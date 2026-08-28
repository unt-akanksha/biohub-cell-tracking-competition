#!/usr/bin/env python
"""Build the frozen, new-window ZSNS005 localization evidence dataset."""

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

from research.temporal_contrastive.verify_division_localization_dataset import (  # noqa: E402
    DATASET_ID,
    RUN_ID,
    SPLIT_CONTRACT,
    V3_V4_VALIDATION_TIMEPOINTS,
    verify_dataset,
)
from research.temporal_contrastive.zebrahub_external import (  # noqa: E402
    ORGANIZER_AUTHORIZATION,
    PUBLIC_ROOT,
    SOURCE_SPECS,
    PublicZarrV2Array,
    build_temporal_patch_shard,
    download_public_file,
    filtered_csv_frames,
    sha256_file,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verified_existing_shard(path: Path, timepoint: int) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"partial localization shard exists: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("source") == "ZSNS005"
        and manifest.get("source_role") == "external_validation"
        and manifest.get("csv_timepoint") == timepoint
        and manifest.get("competition_test_data_read") is False
        and manifest.get("leaderboard_used") is False
        and manifest.get("shard", {}).get("path") == path.name
        and manifest.get("shard", {}).get("bytes") == path.stat().st_size
        and manifest.get("shard", {}).get("sha256") == sha256_file(path)
    ):
        raise ValueError(f"existing localization shard failed verification: {path.name}")
    return manifest


def build_split(
    name: str,
    *,
    cache_root: Path,
    output_root: Path,
    rows: dict[int, dict[str, Any]],
    tracks: dict[str, Any],
    reader: PublicZarrV2Array,
) -> dict[str, Any]:
    timepoints = SPLIT_CONTRACT[name]
    records: list[dict[str, Any]] = []
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    for index, timepoint in enumerate(timepoints, start=1):
        path = target / f"ZSNS005-t{timepoint:04d}.npz"
        manifest = verified_existing_shard(path, timepoint)
        if manifest is None:
            manifest = build_temporal_patch_shard(
                "ZSNS005",
                csv_timepoint=timepoint,
                cache_root=cache_root,
                output_path=path,
                seed=75_005 + timepoint,
                max_sources=64,
                max_targets=96,
                preloaded_rows=rows,
                tracks_record=tracks,
                zarr_reader=reader,
            )
        manifest_path = path.with_suffix(".manifest.json")
        records.append(
            {
                "path": f"{name}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "manifest_path": f"{name}/{manifest_path.name}",
                "manifest_sha256": sha256_file(manifest_path),
                "csv_timepoint": timepoint,
            }
        )
        print(
            json.dumps(
                {
                    "split": name,
                    "completed": index,
                    "required": len(timepoints),
                    "csv_timepoint": timepoint,
                    "shard_sha256": manifest["shard"]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    inventory = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source": "ZSNS005",
        "source_role": "external_validation",
        "split": name,
        "selected_timepoints": list(timepoints),
        "inventory_sha256": inventory,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / ".biohub/cache/external/zebrahub",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".biohub/staging/biohub-division-localization-shards-v1",
    )
    args = parser.parse_args()
    frozen = tuple(SPLIT_CONTRACT["selection"] + SPLIT_CONTRACT["audit"])
    if len(frozen) != 16 or len(set(frozen)) != 16:
        raise ValueError("localization windows repeat or changed count")
    if set(frozen) & V3_V4_VALIDATION_TIMEPOINTS:
        raise ValueError("localization windows overlap v3/v4")

    spec = SOURCE_SPECS["ZSNS005"]
    source_root = args.cache_root / "ZSNS005"
    tracks_url = f"{PUBLIC_ROOT}/ZSNS005_tracks.csv"
    tracks_path = source_root / "ZSNS005_tracks.csv"
    tracks = download_public_file(
        tracks_url, tracks_path, expected_bytes=int(spec["tracks_bytes"])
    )
    required_times = sorted({value for timepoint in frozen for value in (timepoint, timepoint + 1)})
    rows = filtered_csv_frames(tracks_path, required_times)
    reader = PublicZarrV2Array(
        f"{PUBLIC_ROOT}/ZSNS005.ome.zarr/1", source_root / "level1"
    )
    selection = build_split(
        "selection",
        cache_root=args.cache_root,
        output_root=args.output_root,
        rows=rows,
        tracks=tracks,
        reader=reader,
    )
    audit = build_split(
        "audit",
        cache_root=args.cache_root,
        output_root=args.output_root,
        rows=rows,
        tracks=tracks,
        reader=reader,
    )
    source_paths = (
        ROOT / "research/temporal_contrastive/zebrahub_external.py",
        ROOT / "research/temporal_contrastive/transition_context.py",
        ROOT / "research/temporal_contrastive/patch_model.py",
        ROOT / "research/temporal_contrastive/verify_zebrahub_contextual_dataset.py",
        ROOT / "research/temporal_contrastive/verify_division_localization_dataset.py",
    )
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
        "selection_and_audit_frozen_before_extraction": True,
        "disjoint_from_v3_v4_validation": True,
        "selection": selection,
        "audit": audit,
        "sources": {
            path.name: {"path": path.name, "sha256": sha256_file(path)}
            for path in source_paths
        },
    }
    write_json(args.output_root / "DATASET_MANIFEST.json", manifest)
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Division Localization Shards v1",
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

