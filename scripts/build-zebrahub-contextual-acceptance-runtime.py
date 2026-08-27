from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-zebrahub-contextual-acceptance-runtime-v1"
)
RUN_ID = "zebrahub-contextual-acceptance-runtime-v1"
DATASET_ID = f"indarkarhana/biohub-{RUN_ID}"
SOURCES = {
    "contextual_pair_fusion.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "contextual_pair_fusion.py",
    "evaluate_zebrahub_contextual_acceptance.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "evaluate_zebrahub_contextual_acceptance.py",
    "hybrid_linker.py": ROOT
    / "research"
    / "trackastra_graph"
    / "hybrid_linker.py",
    "model.py": ROOT / "research" / "temporal_contrastive" / "model.py",
    "pair_fusion.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "pair_fusion.py",
    "patch_model.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "patch_model.py",
    "submission_sharding.py": ROOT / "research" / "submission_sharding.py",
    "synthetic_data.py": ROOT / "research" / "synthetic_pretrain" / "data.py",
    "train_dual_fold_patch.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_dual_fold_patch.py",
    "train_zebrahub_contextual_pretrain.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_zebrahub_contextual_pretrain.py",
    "trainer.py": ROOT
    / "research"
    / "trackastra_graph"
    / "train_biohub_graph_transformer.py",
    "transition_context.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "transition_context.py",
    "verify_runtime.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "verify_runtime.py",
    "verify_zebrahub_contextual_acceptance.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "verify_zebrahub_contextual_acceptance.py",
    "verify_zebrahub_contextual_dataset.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "verify_zebrahub_contextual_dataset.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_target() -> Path:
    target = TARGET.resolve()
    staging = (ROOT / ".biohub" / "staging").resolve()
    if target.parent != staging or target.name != TARGET.name:
        raise RuntimeError(f"unsafe acceptance runtime target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = checked_target()
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} exists; pass --replace to rebuild")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"acceptance runtime sources are missing: {missing}")
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(target.iterdir())
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "runtime_family": "contextual_pair_fusion_v3_external_acceptance",
            "purpose": (
                "one-shot, two-GPU evaluation of hash-bound contextual v3 "
                "checkpoints on frozen ZSNS001; no competition or submission path"
            ),
            "appearance_model": {
                "family": "temporal_contextual_pair_fusion_v3",
                "parameters_per_fold": 20_747_761,
                "folds": ["target_44b6", "target_6bba"],
                "acceptance_source": "ZSNS001",
                "acceptance_shards": 16,
                "acceptance_manifest_sha256": (
                    "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"
                ),
                "acceptance_inventory_sha256": (
                    "e32bc686e14222e43acb8d6247351e286eae8ed6fdb1f4ab5087e55fb0c79667"
                ),
                "acceptance_record_inventory_sha256": (
                    "05ad8b3195aa2786ffb8a2ffcb247d118d1e5cf96026264e0534c468979e6e0e"
                ),
            },
            "integrity": {
                "required_gpu_count": 2,
                "opened_processed_acceptance_stems_excluded_from_training": True,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "competition_submission_command_included": False,
                "calibration_grid_includes_exact_zero_control": True,
                "maximum_submission_inference_seconds": 36_000,
                "minimum_kaggle_finalization_reserve_seconds": 7_200,
                "required_kaggle_machine_shape": "NvidiaTeslaT4",
                "submission_internet_enabled": False,
                "timed_out_worker_termination_grace_seconds": 15,
                "predeclared_trackastra_control_allowed": True,
                "one_shot_acceptance_hard_stop_seconds": 3_600,
                "competition_data_read": False,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub ZebraHub Contextual Acceptance Runtime v1",
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "source_files": len(files),
                "bytes": sum(
                    path.stat().st_size
                    for path in target.iterdir()
                    if path.is_file()
                ),
                "manifest_sha256": sha256_file(target / "SOURCE_MANIFEST.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
