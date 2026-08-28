#!/usr/bin/env python
"""Build the final contextual-v3 guarded-launch preflight for Kaggle outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from biohub_tracker.io import sha256_file  # noqa: E402
from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)


RUN_ID = "temporal-contextual-pair-fusion-candidate-v3"
KERNEL_ID = "biohub-temporal-contextual-submission-candidate-v3"
KERNEL_DIR = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = KERNEL_DIR / f"{KERNEL_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
KERNEL_BUILDER = (
    ROOT / "scripts" / "build-temporal-contextual-submission-candidate-kernel.py"
)
PREFLIGHT_BUILDER = Path(__file__).resolve()
COMPETITION_CONFIG = ROOT / "config" / "competition.json"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json"
APPEARANCE_KERNEL_ID = (
    "indarkarhana/biohub-temporal-contextual-transfer-v3"
)
ACCEPTANCE_DATASET_ID = (
    "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3"
)
ACCEPTANCE_FILENAME = (
    "temporal-contextual-pair-fusion-v3-exact-acceptance.json"
)
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
FOLDS = {"target_44b6", "target_6bba"}
EXPECTED_DATASET_SOURCES = [
    "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
    ACCEPTANCE_DATASET_ID,
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]
EXPECTED_KERNEL_SOURCES = [
    "indarkarhana/biohub-clean-0-927-reproduction-v1",
    "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    APPEARANCE_KERNEL_ID,
]
EXPECTED_PROCESSED_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
FOCUSED_TESTS = (
    ROOT / "tests" / "test_temporal_contextual_exact_acceptance_runner.py",
    ROOT / "tests" / "test_temporal_contextual_processed_acceptance_kernel_builder.py",
    ROOT / "tests" / "test_temporal_contextual_submission_candidate_kernel_builder.py",
    ROOT / "tests" / "test_temporal_contextual_kernel_submission.py",
    ROOT / "tests" / "test_temporal_appearance_submission.py",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def checked_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"{label} must remain inside the workspace")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def check(name: str, evidence: list[Path], detail: str) -> PreflightCheck:
    return PreflightCheck.create(
        name,
        "passed",
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def verify_manifest_files(root: Path, manifest: dict) -> list[Path]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("appearance artifact manifest has no files")
    verified = []
    for relative_path, evidence in files.items():
        path = root / relative_path
        if not path.is_file() or not isinstance(evidence, dict):
            raise RuntimeError(f"appearance artifact file is missing: {relative_path}")
        if (
            evidence.get("bytes") != path.stat().st_size
            or evidence.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"appearance artifact file changed: {relative_path}")
        verified.append(path)
    return verified


def notebook_policy() -> tuple[str, dict]:
    subprocess.run([sys.executable, str(KERNEL_BUILDER)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    if not (
        metadata.get("id")
        == "indarkarhana/biohub-temporal-contextual-submission-candidate-v3"
        and metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and metadata.get("dataset_sources") == EXPECTED_DATASET_SOURCES
        and metadata.get("kernel_sources") == EXPECTED_KERNEL_SOURCES
        and metadata.get("competition_sources")
        == ["biohub-cell-tracking-during-development"]
    ):
        raise RuntimeError("final contextual kernel input policy changed")
    required = (
        "torch.cuda.device_count() != 2",
        "DECLARED_BUDGET_SECONDS = 43_200",
        "INFERENCE_HARD_STOP_SECONDS = 36_000",
        "FINALIZATION_RESERVE_SECONDS = 7_200",
        'payload.get("status") == "accepted"',
        'payload.get("exact_processed_gate_passed") is True',
        'report.get("nodes_preserved_exactly") is True',
        'report.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256',
        'report.get("competition_submission_performed") is False',
    )
    missing = [fragment for fragment in required if fragment not in code]
    if missing:
        raise RuntimeError(f"final contextual kernel lost required gates: {missing}")
    if "competitions submit" in code.casefold() or "kaggle competitions" in code.casefold():
        raise RuntimeError("final contextual kernel contains a submission command")
    return code, metadata


def verify_staged_inputs(
    appearance_root: Path,
    acceptance_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    appearance_root = checked_path(appearance_root, "Kaggle transfer output")
    acceptance_root = checked_path(acceptance_root, "acceptance dataset")
    processed_root = checked_path(processed_root, "processed Kaggle output")

    training_candidates = []
    for path in appearance_root.rglob("training_terminal.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            payload.get("run_id") == "temporal-contextual-pair-fusion-v3"
            and payload.get("appearance_family") == APPEARANCE_FAMILY
        ):
            training_candidates.append(path)
    if len(training_candidates) != 1:
        raise RuntimeError(
            "Kaggle transfer output has ambiguous training evidence: "
            f"{[str(path) for path in training_candidates]}"
        )
    training_terminal_path = training_candidates[0]
    appearance_training_root = training_terminal_path.parent
    training = json.loads(training_terminal_path.read_text(encoding="utf-8"))
    if not (
        training.get("status") == "completed"
        and training.get("run_id") == "temporal-contextual-pair-fusion-v3"
        and training.get("appearance_family") == APPEARANCE_FAMILY
        and training.get("gpu_count") == 2
        and training.get("both_folds_trained") is True
        and training.get("both_folds_improved") is True
        and set(training.get("folds", {})) == FOLDS
        and training.get("public_predictions_copied") is False
        and training.get("public_leaderboard_used_for_selection") is False
        and training.get("submission_created") is False
    ):
        raise RuntimeError("Kaggle transfer terminal is invalid")
    appearance_files = []
    for fold in FOLDS:
        checkpoint = appearance_training_root / fold / "appearance_model.pt"
        row = training["folds"][fold]
        if not (
            checkpoint.is_file()
            and row.get("status") == "completed"
            and row.get("appearance_family") == APPEARANCE_FAMILY
            and int(row.get("best_step", 0)) > 0
            and row.get("finetuning_gate_passed") is True
            and row.get("model_sha256") == sha256_file(checkpoint)
        ):
            raise RuntimeError(f"Kaggle transfer checkpoint is invalid: {fold}")
        appearance_files.append(checkpoint)
    transfer_launchers = []
    for path in appearance_root.rglob("launcher_terminal.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("run_id") == "temporal-contextual-pair-fusion-v3":
            transfer_launchers.append((path, payload))
    if len(transfer_launchers) != 1:
        raise RuntimeError("Kaggle transfer launcher evidence is ambiguous")
    transfer_launcher_path, transfer_launcher = transfer_launchers[0]
    if not (
        transfer_launcher.get("schema_version") == 1
        and transfer_launcher.get("run_id")
        == "temporal-contextual-pair-fusion-v3"
        and transfer_launcher.get("status") == "completed"
        and transfer_launcher.get("gpu_count_required") == 2
        and transfer_launcher.get("declared_budget_seconds") == 39_600
        and transfer_launcher.get("trainer_max_wall_seconds") == 36_000
        and transfer_launcher.get("trainer_hard_stop_seconds") == 37_800
        and transfer_launcher.get("training_terminal_exists") is True
        and transfer_launcher.get("training_terminal_sha256")
        == sha256_file(training_terminal_path)
        and transfer_launcher.get("public_predictions_copied") is False
        and transfer_launcher.get("public_leaderboard_used_for_selection") is False
        and transfer_launcher.get("submission_created") is False
    ):
        raise RuntimeError("Kaggle transfer launcher evidence is invalid")

    acceptance_metadata = json.loads(
        (acceptance_root / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    acceptance_manifest_path = acceptance_root / "ARTIFACT_MANIFEST.json"
    acceptance_manifest = json.loads(
        acceptance_manifest_path.read_text(encoding="utf-8")
    )
    acceptance_path = acceptance_root / ACCEPTANCE_FILENAME
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not (
        acceptance_metadata.get("id") == ACCEPTANCE_DATASET_ID
        and acceptance_metadata.get("isPrivate") is True
        and acceptance_manifest.get("artifact_kind")
        == "contextual_v3_exact_acceptance"
        and acceptance_manifest.get("acceptance_evidence", {}).get("sha256")
        == sha256_file(acceptance_path)
        and acceptance.get("status") == "accepted"
        and acceptance.get("exact_processed_gate_passed") is True
        and acceptance.get("candidate_family") == CANDIDATE_FAMILY
        and acceptance.get("appearance_family") == APPEARANCE_FAMILY
        and acceptance.get("candidate_node_rows_identical") is True
        and acceptance.get("candidate_edge_sets_differ") is True
        and acceptance.get("public_leaderboard_used_for_selection") is False
        and acceptance.get("competition_submission_performed") is False
        and set(acceptance.get("appearance_models", {})) == FOLDS
    ):
        raise RuntimeError("staged exact acceptance dataset is invalid")
    for fold in FOLDS:
        if acceptance["appearance_models"][fold].get("model_sha256") != training[
            "folds"
        ][fold].get("model_sha256"):
            raise RuntimeError(f"staged accepted checkpoint mismatch: {fold}")

    materialization_candidates = []
    for path in processed_root.rglob("materialization_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            payload.get("run_id")
            == "temporal-contextual-pair-fusion-processed-acceptance-v3"
            and payload.get("appearance_family") == APPEARANCE_FAMILY
        ):
            materialization_candidates.append((path, payload))
    if len(materialization_candidates) != 1:
        raise RuntimeError("processed Kaggle materialization evidence is ambiguous")
    materialization_path, materialization = materialization_candidates[0]
    candidate_path = materialization_path.with_name("processed_candidate.csv")
    processed_launcher_candidates = []
    for path in processed_root.rglob("processed_launcher_terminal.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            payload.get("run_id")
            == "temporal-contextual-pair-fusion-processed-acceptance-v3"
        ):
            processed_launcher_candidates.append((path, payload))
    if len(processed_launcher_candidates) != 1:
        raise RuntimeError("processed Kaggle launcher evidence is ambiguous")
    processed_launcher_path, processed_launcher = processed_launcher_candidates[0]
    datasets = materialization.get("datasets", {})
    if not (
        candidate_path.is_file()
        and processed_launcher.get("schema_version") == 1
        and processed_launcher.get("run_id")
        == "temporal-contextual-pair-fusion-processed-acceptance-v3"
        and processed_launcher.get("status") == "completed"
        and processed_launcher.get("declared_budget_seconds") == 21_600
        and processed_launcher.get("materializer_hard_stop_seconds") == 19_800
        and processed_launcher.get("gpu_count_required") == 2
        and processed_launcher.get("whole_movie_sharding_required") is True
        and processed_launcher.get("materialization_result_exists") is True
        and processed_launcher.get("processed_candidate_exists") is True
        and processed_launcher.get("result_sha256")
        == sha256_file(materialization_path)
        and processed_launcher.get("candidate_sha256") == sha256_file(candidate_path)
        and processed_launcher.get("processed_ground_truth_read") is False
        and processed_launcher.get("exact_processed_scoring_performed") is False
        and processed_launcher.get("public_leaderboard_used_for_selection") is False
        and processed_launcher.get("competition_submission_performed") is False
        and processed_launcher.get("authorized_for_submission") is False
        and materialization.get("schema_version") == 1
        and materialization.get("status") == "completed"
        and materialization.get("evaluation_kind")
        == "predeclared_processed_candidate_materialization"
        and materialization.get("candidate_family") == CANDIDATE_FAMILY
        and materialization.get("appearance_family") == APPEARANCE_FAMILY
        and materialization.get("gpu_count") == 2
        and materialization.get("whole_movie_sharding") is True
        and materialization.get("processed_candidate_sha256")
        == sha256_file(candidate_path)
        and int(materialization.get("total_changed_edges", 0)) > 0
        and isinstance(datasets, dict)
        and set(datasets) == EXPECTED_PROCESSED_STEMS
        and materialization.get("ground_truth_read") is False
        and materialization.get("hyperparameter_selection_performed") is False
        and materialization.get("exact_processed_scoring_performed") is False
        and materialization.get("public_leaderboard_used_for_selection") is False
        and materialization.get("competition_submission_performed") is False
        and materialization.get("authorized_for_submission") is False
        and acceptance.get("processed_candidate_sha256")
        == sha256_file(candidate_path)
        and acceptance.get("materialization_result_sha256")
        == sha256_file(materialization_path)
    ):
        raise RuntimeError("processed Kaggle benchmark evidence is invalid")
    for fold in FOLDS:
        if materialization.get("appearance_models", {}).get(fold, {}).get(
            "model_sha256"
        ) != training["folds"][fold].get("model_sha256"):
            raise RuntimeError(f"processed checkpoint mismatch: {fold}")
    return {
        "appearance_files": appearance_files,
        "training_terminal": training_terminal_path,
        "transfer_launcher": transfer_launcher_path,
        "acceptance_manifest": acceptance_manifest_path,
        "acceptance_evidence": acceptance_path,
        "processed_launcher": processed_launcher_path,
        "materialization_result": materialization_path,
        "processed_candidate": candidate_path,
    }


def main(
    *,
    appearance_root: Path,
    acceptance_root: Path,
    processed_root: Path,
    runtime_root: Path,
    output: Path,
) -> None:
    code, _metadata = notebook_policy()
    staged = verify_staged_inputs(
        appearance_root, acceptance_root, processed_root
    )
    runtime_root = checked_path(runtime_root, "runtime")
    subprocess.run(
        [
            sys.executable,
            str(runtime_root / "verify_appearance_output.py"),
            "--root",
            str(appearance_root),
            "--expected-family",
            APPEARANCE_FAMILY,
            "--strict-checkpoint",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[relative(path) for path in FOCUSED_TESTS],
        ],
        cwd=ROOT,
        check=True,
    )
    checks = [
        check(
            "imports",
            [PREFLIGHT_BUILDER, KERNEL_BUILDER, NOTEBOOK, *FOCUSED_TESTS],
            "The final notebook compiles and the complete candidate transport/upload boundary suite passes.",
        ),
        check(
            "inputs",
            [
                METADATA,
                staged["transfer_launcher"],
                staged["acceptance_manifest"],
                staged["training_terminal"],
                staged["acceptance_evidence"],
            ],
            "Only the exact runtime, Kaggle transfer output, private exact acceptance, support, base, Trackastra, and competition sources are attached.",
        ),
        check(
            "single_batch",
            [
                ROOT / "tests" / "test_temporal_appearance_submission.py",
                runtime_root / "dual_fold_appearance_submission.py",
            ],
            "Contextual-v3 acceptance loading, family dispatch, two-GPU sharding, and candidate construction have focused finite fixtures.",
        ),
        PreflightCheck.create(
            "model_step",
            "not_applicable",
            [],
            ROOT,
            detail="Final inference is frozen and performs no optimizer step.",
        ),
        check(
            "checkpoint_roundtrip",
            [
                staged["training_terminal"],
                *[
                    appearance_root / fold / "appearance_model.pt"
                    for fold in sorted(FOLDS)
                ],
                runtime_root / "verify_appearance_output.py",
            ],
            "Both accepted contextual-v3 checkpoints strict-load against the exact cloud terminal and acceptance hashes.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA, staged["processed_launcher"]],
            "The notebook writes submission.csv and evidence below /kaggle/working and contains no upload command.",
        ),
        check(
            "dense_memory",
            [staged["processed_launcher"], staged["materialization_result"]],
            "The exact two-GPU whole-movie materializer completed on Kaggle under its frozen hard stop before final test inference.",
        ),
        check(
            "dataset_coverage",
            [staged["materialization_result"], staged["processed_candidate"]],
            "The Kaggle benchmark covers all four predeclared complete movies, changes edges, and is hash-bound to exact accepted scoring.",
        ),
        check(
            "non_replica_provenance",
            [
                staged["acceptance_evidence"],
                staged["candidate_report"],
                ROOT / "reports" / "experiments" / "biohub-public-frontier-refresh-2026-08-27.md",
            ],
            "Project-authored contextual evidence passed the exact clean gate; metric hacks, public predictions, and leaderboard selection remain excluded.",
        ),
        check(
            "quota_policy",
            [COMPETITION_CONFIG, NOTEBOOK, staged["processed_launcher"]],
            "The final launch still requires the live account-wide reserve guard and exactly two Kaggle GPUs.",
        ),
    ]
    if "torch.cuda.device_count() != 2" not in code:
        raise RuntimeError("two-GPU final-inference gate disappeared")
    report = PreflightReport.create(RUN_ID, checks)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(output, report)
    print(output)
    print(report.report_sha256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(
        appearance_root=args.appearance_root,
        acceptance_root=args.acceptance_root,
        processed_root=args.processed_root,
        runtime_root=args.runtime_root,
        output=args.output,
    )
