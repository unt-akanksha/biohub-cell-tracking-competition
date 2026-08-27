from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET_BY_FAMILY = {
    "cosine_v1": STAGING_ROOT / "biohub-temporal-patch-runtime-v1",
    "pair_fusion_v2": STAGING_ROOT / "biohub-temporal-pair-fusion-runtime-v2",
    "contextual_pair_fusion_v3": (
        STAGING_ROOT / "biohub-temporal-contextual-pair-fusion-runtime-v3"
    ),
}
RUN_ID_BY_FAMILY = {
    "cosine_v1": "temporal-patch-dual-fold-v1",
    "pair_fusion_v2": "temporal-patch-pair-fusion-v2",
    "contextual_pair_fusion_v3": "temporal-contextual-pair-fusion-v3",
}
DATASET_ID_BY_FAMILY = {
    "cosine_v1": "indarkarhana/biohub-temporal-patch-runtime-v1",
    "pair_fusion_v2": "indarkarhana/biohub-temporal-pair-fusion-runtime-v2",
    "contextual_pair_fusion_v3": (
        "indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3"
    ),
}
EXPERIMENT_BY_FAMILY = {
    "cosine_v1": ROOT
    / "config"
    / "experiments"
    / "temporal-patch-dual-fold-v1.json",
    "pair_fusion_v2": ROOT
    / "config"
    / "experiments"
    / "temporal-patch-pair-fusion-v2.json",
    "contextual_pair_fusion_v3": ROOT
    / "config"
    / "experiments"
    / "temporal-contextual-pair-fusion-v3.json",
}
TITLE_BY_FAMILY = {
    "cosine_v1": "Biohub Temporal Patch Runtime v1",
    "pair_fusion_v2": "Biohub Temporal Pair Fusion Runtime v2",
    "contextual_pair_fusion_v3": (
        "Biohub Temporal Contextual Pair Fusion Runtime v3"
    ),
}
TRACKASTRA_REPOSITORY = ROOT / ".biohub" / "cache" / "repos" / "trackastra"
EXPECTED_TRACKASTRA_COMMIT = "6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b"
SOURCES = {
    "model.py": ROOT / "research" / "temporal_contrastive" / "model.py",
    "patch_model.py": ROOT / "research" / "temporal_contrastive" / "patch_model.py",
    "pair_fusion.py": ROOT / "research" / "temporal_contrastive" / "pair_fusion.py",
    "transition_context.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "transition_context.py",
    "contextual_pair_fusion.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "contextual_pair_fusion.py",
    "contextual_training.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "contextual_training.py",
    "appearance_family.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "appearance_family.py",
    "train_dual_fold_patch.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_dual_fold_patch.py",
    "train_dual_fold_pair_fusion.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_dual_fold_pair_fusion.py",
    "train_dual_fold_contextual_pair_fusion.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_dual_fold_contextual_pair_fusion.py",
    "train_zebrahub_contextual_pretrain.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "train_zebrahub_contextual_pretrain.py",
    "appearance_blend.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "appearance_blend.py",
    "calibrate_dual_fold_blend.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "calibrate_dual_fold_blend.py",
    "dual_fold_appearance_processed_acceptance.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "dual_fold_appearance_processed_acceptance.py",
    "dual_fold_appearance_submission.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "dual_fold_appearance_submission.py",
    "verify_runtime.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "verify_runtime.py",
    "verify_appearance_output.py": ROOT
    / "research"
    / "temporal_contrastive"
    / "verify_dual_fold_training_output.py",
    "verify_trackastra_output.py": ROOT
    / "research"
    / "trackastra_graph"
    / "verify_dual_fold_training_output.py",
    "trainer.py": ROOT
    / "research"
    / "trackastra_graph"
    / "train_biohub_graph_transformer.py",
    "synthetic_data.py": ROOT / "research" / "synthetic_pretrain" / "data.py",
    "hybrid_linker.py": ROOT / "research" / "trackastra_graph" / "hybrid_linker.py",
    "rerank_submission.py": ROOT
    / "research"
    / "trackastra_graph"
    / "rerank_submission.py",
    "dual_fold_processed_acceptance.py": ROOT
    / "research"
    / "trackastra_graph"
    / "dual_fold_processed_acceptance.py",
    "dual_fold_rerank_submission.py": ROOT
    / "research"
    / "trackastra_graph"
    / "dual_fold_rerank_submission.py",
    "submission_sharding.py": ROOT / "research" / "submission_sharding.py",
}
TRACKASTRA_FILES = (
    "trackastra/__init__.py",
    "trackastra/_version.py",
    "trackastra/model/__init__.py",
    "trackastra/model/model.py",
    "trackastra/model/model_parts.py",
    "trackastra/model/rope.py",
    "trackastra/utils/__init__.py",
    "trackastra/utils/utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_target(family: str) -> Path:
    staging = STAGING_ROOT.resolve()
    target = TARGET_BY_FAMILY[family].resolve()
    if target.parent != staging or target not in {
        path.resolve() for path in TARGET_BY_FAMILY.values()
    }:
        raise RuntimeError(f"unsafe temporal runtime target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--family", choices=sorted(TARGET_BY_FAMILY), default="cosine_v1"
    )
    args = parser.parse_args()
    target = checked_target(args.family)
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} exists; pass --replace to rebuild")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    commit = subprocess.check_output(
        ["git", "-C", str(TRACKASTRA_REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_TRACKASTRA_COMMIT:
        raise RuntimeError(f"Trackastra source commit changed: {commit}")
    sources = {**SOURCES, "experiment.json": EXPERIMENT_BY_FAMILY[args.family]}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"temporal runtime sources are missing: {missing}")
    for name, source in sources.items():
        shutil.copy2(source, target / name)
    shutil.copy2(TRACKASTRA_REPOSITORY / "LICENSE", target / "TRACKASTRA_LICENSE")
    for relative in TRACKASTRA_FILES:
        source = TRACKASTRA_REPOSITORY / relative
        destination = target / "trackastra_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID_BY_FAMILY[args.family],
            "runtime_family": args.family,
            "purpose": "portable two-GPU external pretraining and reciprocal appearance training, clean calibration, exact processed materialization, and candidate building; no submit command",
            "trackastra": {
                "repository": "https://github.com/weigertlab/trackastra",
                "commit": commit,
                "license": "BSD-3-Clause",
            },
            "appearance_model": {
                "implementation": "independent Biohub physical-scale 3D residual encoder with optional learned candidate-pair and project-authored transition-context fusion",
                "families": {
                    "temporal_cosine_v1": {"parameters_per_fold": 19_221_954},
                    "temporal_pair_fusion_v2": {
                        "parameters_per_fold": 20_869_325,
                        "pair_feature_width": 1_029,
                        "pair_projection_width": 1_024,
                        "pair_hidden_widths": [512, 128],
                    },
                    "temporal_contextual_pair_fusion_v3": {
                        "parameters_per_fold": 20_747_761,
                        "candidate_context_width": 18,
                        "contextual_pair_feature_width": 1_047,
                        "edge_token_width": 256,
                        "edge_set_feature_width": 1_536,
                    },
                },
                "input_channels": 3,
                "temporal_frame_offsets": [-1, 0, 1],
                "external_pretrained_weights_required_by_family": {
                    "temporal_cosine_v1": False,
                    "temporal_pair_fusion_v2": False,
                    "temporal_contextual_pair_fusion_v3": True,
                },
                "division_head_used_at_inference": True,
                "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
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
                "contextual_v3_requires_hash_bound_external_pretraining": True,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": TITLE_BY_FAMILY[args.family],
            "id": DATASET_ID_BY_FAMILY[args.family],
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "source_files": len(files),
                "bytes": sum(
                    path.stat().st_size for path in target.rglob("*") if path.is_file()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
