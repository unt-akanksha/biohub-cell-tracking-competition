#!/usr/bin/env python
"""Extract parent/daughter relational division patches on Kaggle CPU only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np
import torch


RUN_ID = "competition-relational-division-patches-v3"
INVENTORY_RUN_ID = "competition-relational-division-inventory-v3"
INVENTORY_DATASET_RUN_ID = "competition-relational-division-inventory-dataset-v3"
INVENTORY_SHA256 = "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
INVENTORY_MANIFEST_SHA256 = "c5f093601d74201739e003f95eb56511b33df472d4e5683f7354584bd086ee23"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
CENTER_NAMES = ("parent", "existing_child", "proposed_child")
GEOMETRY_NAMES = (
    "parent_distance_um",
    "sister_distance_um",
    "existing_distance_um",
    "daughter_midpoint_distance_um",
    "daughter_opposition_cosine",
    "daughter_step_ratio",
    "biological_geometry_score",
    "parent_velocity_um",
    "constant_velocity_midpoint_error_um",
)


def _load_base_module() -> Any:
    candidates = [
        Path(
            "/kaggle/input/datasets/indarkarhana/"
            "biohub-real-division-extractor-runtime-v1/extract_base.py"
        ),
        Path("/kaggle/input/biohub-real-division-extractor-runtime-v1/extract_base.py"),
        Path(__file__).resolve().parents[1]
        / "biohub-real-division-patches-v1"
        / "extract.py",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError({"eligible_extractor_bases": [str(path) for path in matches]})
    spec = importlib.util.spec_from_file_location(
        "biohub_relational_division_extract_base", matches[0]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned real-division extractor base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def find_inventory(input_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [
        input_root
        / "datasets/indarkarhana/biohub-relational-division-inventory-v3",
        input_root / "biohub-relational-division-inventory-v3",
    ]
    candidates.extend(path for path in input_root.iterdir() if path.is_dir())
    matches = []
    for root in dict.fromkeys(candidates):
        manifest_path = root / "RELATIONAL_DIVISION_INVENTORY_MANIFEST.json"
        inventory_path = root / "relational_division_inventory_v3.json"
        if not manifest_path.is_file() or not inventory_path.is_file():
            continue
        if (
            base.sha256_file(manifest_path) != INVENTORY_MANIFEST_SHA256
            or base.sha256_file(inventory_path) != INVENTORY_SHA256
        ):
            continue
        matches.append((inventory_path, json.loads(inventory_path.read_text())))
    if len(matches) != 1:
        raise RuntimeError({"eligible_relational_inventories": [str(row[0]) for row in matches]})
    return matches[0]


def validate_inventory(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    examples = payload.get("examples", [])
    by_role = summary.get("by_embryo_role", {})
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and payload.get("run_id") == INVENTORY_RUN_ID
        and set(payload.get("final_probe_stems", [])) == FINAL_PROBE_STEMS
        and payload.get("audit_opened") is False
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_code_copied") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and summary.get("rows") == len(examples) == 3_013
        and summary.get("positives") == 134
        and summary.get("hard_negatives") == 2_879
        and summary.get("inference_eligible_positives") == 55
        and summary.get("inference_eligible_hard_negatives") == 165
        and all(
            by_role.get(embryo, {}).get(role, {}).get("positives", 0) > 0
            and by_role.get(embryo, {}).get(role, {}).get("hard_negatives", 0) > 0
            and by_role.get(embryo, {})
            .get(role, {})
            .get("inference_eligible_positives", 0)
            > 0
            and by_role.get(embryo, {})
            .get(role, {})
            .get("inference_eligible_hard_negatives", 0)
            > 0
            for embryo in ("44b6", "6bba")
            for role in ("optimization", "selection", "audit")
        )
    ):
        raise ValueError("relational division inventory changed")
    seen = set()
    for row in examples:
        key = (
            row.get("stem"),
            row.get("parent_id"),
            row.get("existing_child_id"),
            row.get("proposed_child_id"),
        )
        if not (
            row.get("stem") not in FINAL_PROBE_STEMS
            and row.get("role") in {"optimization", "selection", "audit"}
            and row.get("embryo") in {"44b6", "6bba"}
            and row.get("daughter_order_invariant") is True
            and isinstance(row.get("division_recovery_target"), bool)
            and isinstance(row.get("inference_geometry_eligible"), bool)
            and set(row.get("centers_zyx_voxel", {})) == set(CENTER_NAMES)
            and len(row.get("required_frames", [])) == 3
            and key not in seen
        ):
            raise ValueError(f"relational division example changed: {key}")
        seen.add(key)


def write_shard(
    output_root: Path,
    *,
    stem: str,
    role: str,
    embryo: str,
    timepoint: int,
    examples: list[dict[str, Any]],
    patches: torch.Tensor,
) -> dict[str, Any]:
    expected_shape = (len(examples), len(CENTER_NAMES), 3, 17, 17, 17)
    if tuple(patches.shape) != expected_shape:
        raise ValueError(f"relational patch shape changed: {tuple(patches.shape)}")
    target = output_root / role / embryo / f"{stem}-t{timepoint:03d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [row["division_recovery_target"] for row in examples], dtype=np.float32
    )
    eligible = np.asarray(
        [row["inference_geometry_eligible"] for row in examples], dtype=np.bool_
    )
    weights = np.where(labels > 0.5, 1.0, np.where(eligible, 1.0, 0.50)).astype(
        np.float32
    )
    geometry = np.asarray(
        [
            [
                np.nan if row.get(name) is None else float(row[name])
                for name in GEOMETRY_NAMES
            ]
            for row in examples
        ],
        dtype=np.float32,
    )
    metadata = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stem": stem,
        "embryo": embryo,
        "role": role,
        "timepoint": timepoint,
        "rows": len(examples),
        "positives": int(labels.sum()),
        "hard_negatives": int((labels < 0.5).sum()),
        "inference_eligible_positives": int(np.sum((labels > 0.5) & eligible)),
        "inference_eligible_hard_negatives": int(
            np.sum((labels < 0.5) & eligible)
        ),
        "patch_layout": "N,parent-existing-proposed,temporal-3,Z17,Y17,X17",
        "geometry_names": list(GEOMETRY_NAMES),
        "daughter_order_invariant": True,
        "audit_labels_scored": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
    }
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            relational_patches=patches.numpy().astype(np.float16),
            geometry_features=geometry,
            division_recovery_target=labels,
            label_weight=weights,
            inference_geometry_eligible=eligible,
            parent_ids=np.asarray([row["parent_id"] for row in examples]),
            existing_child_ids=np.asarray(
                [row["existing_child_id"] for row in examples]
            ),
            proposed_child_ids=np.asarray(
                [row["proposed_child_id"] for row in examples]
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(target)
    return {
        **metadata,
        "path": target.relative_to(output_root).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": base.sha256_file(target),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "shards": len(records),
        "movies": len({row["stem"] for row in records}),
        "rows": sum(row["rows"] for row in records),
        "positives": sum(row["positives"] for row in records),
        "hard_negatives": sum(row["hard_negatives"] for row in records),
        "inference_eligible_positives": sum(
            row["inference_eligible_positives"] for row in records
        ),
        "inference_eligible_hard_negatives": sum(
            row["inference_eligible_hard_negatives"] for row in records
        ),
    }


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("relational division extraction must use Kaggle CPU only")
    started = time.monotonic()
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    base.install_numcodecs(input_root, working)
    inventory_path, inventory = find_inventory(input_root)
    validate_inventory(inventory)
    train_root = base.discover_train_root(input_root)
    output_root = working / "biohub_relational_division_patches_v3"
    output_root.mkdir(parents=True, exist_ok=False)

    by_stem: dict[str, list[dict[str, Any]]] = {}
    for row in inventory["examples"]:
        by_stem.setdefault(str(row["stem"]), []).append(row)
    records = []
    for movie_index, (stem, movie_rows) in enumerate(sorted(by_stem.items()), start=1):
        image_path = train_root / f"{stem}.zarr" / "0"
        if not image_path.is_dir():
            raise FileNotFoundError(f"relational movie image is missing: {stem}")
        for timepoint in sorted({int(row["timepoint"]) for row in movie_rows}):
            selected = [
                row for row in movie_rows if int(row["timepoint"]) == timepoint
            ]
            context = np.stack(
                [base.read_v3_frame(image_path, index) for index in selected[0]["required_frames"]]
            )
            centers = np.asarray(
                [
                    row["centers_zyx_voxel"][name]
                    for row in selected
                    for name in CENTER_NAMES
                ],
                dtype=np.float32,
            )
            sampled = base.sample_physical_patches(context, centers)
            patches = sampled.reshape(len(selected), len(CENTER_NAMES), *sampled.shape[1:])
            records.append(
                write_shard(
                    output_root,
                    stem=stem,
                    role=str(selected[0]["role"]),
                    embryo=str(selected[0]["embryo"]),
                    timepoint=timepoint,
                    examples=selected,
                    patches=patches,
                )
            )
        if movie_index % 20 == 0:
            print(f"processed {movie_index}/{len(by_stem)} relational movies", flush=True)

    summary = summarize(records)
    if not (
        summary["rows"] == inventory["summary"]["rows"]
        and summary["positives"] == inventory["summary"]["positives"]
        and summary["hard_negatives"] == inventory["summary"]["hard_negatives"]
        and summary["inference_eligible_positives"]
        == inventory["summary"]["inference_eligible_positives"]
        and summary["inference_eligible_hard_negatives"]
        == inventory["summary"]["inference_eligible_hard_negatives"]
    ):
        raise RuntimeError("relational extracted inventory does not reproduce source")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "elapsed_seconds": time.monotonic() - started,
        "inventory_sha256": base.sha256_file(inventory_path),
        "records": records,
        "summary": summary,
        "audit_features_extracted": True,
        "audit_labels_scored": False,
        "final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "final_probe_movies_extracted": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    base.atomic_json(output_root / "relational_division_patch_manifest.json", manifest)
    archive_path = working / "biohub_relational_division_patches_v3.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(output_root, arcname=output_root.name)
    print(
        json.dumps(
            {
                "manifest": str(output_root / "relational_division_patch_manifest.json"),
                "manifest_sha256": base.sha256_file(
                    output_root / "relational_division_patch_manifest.json"
                ),
                "archive": str(archive_path),
                "archive_sha256": base.sha256_file(archive_path),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
