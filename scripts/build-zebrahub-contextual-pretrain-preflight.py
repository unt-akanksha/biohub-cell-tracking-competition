from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import torch


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
from research.temporal_contrastive.contextual_pair_fusion import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    RECIPROCAL_PARENT_LOSS_WEIGHT,
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (  # noqa: E402
    VALIDATION_PARTITION_POLICY,
    discover_shards,
    load_shard,
    partition_validation_records,
)


RUN_ID = "zebrahub-contextual-pretrain-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
KERNEL_BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-pretrain-kernel.py"
PREFLIGHT_BUILDER = Path(__file__).resolve()
RUNTIME = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-temporal-contextual-pair-fusion-runtime-v3-version8"
)
DATASET = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-zebrahub-contextual-shards-v1-version4"
)
RUNTIME_MANIFEST = RUNTIME / "SOURCE_MANIFEST.json"
DATASET_MANIFEST = DATASET / "DATASET_MANIFEST.json"
CONFIG = ROOT / "config" / "experiments" / "temporal-contextual-pair-fusion-v3.json"
COMPETITION_CONFIG = ROOT / "config" / "competition.json"
SMOKE = ROOT / "reports" / "experiments" / "zebrahub-contextual-reciprocal-smoke.json"
TECHNIQUE_REPORT = ROOT / "reports" / "experiments" / "biohub-technique-scan-2026-08-27.md"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json"
FOCUSED_TESTS = (
    ROOT / "tests" / "test_contextual_pair_fusion.py",
    ROOT / "tests" / "test_contextual_training.py",
    ROOT / "tests" / "test_contextual_trainer.py",
    ROOT / "tests" / "test_submission_sharding.py",
    ROOT / "tests" / "test_temporal_appearance_processed_acceptance.py",
    ROOT / "tests" / "test_temporal_appearance_submission.py",
    ROOT / "tests" / "test_temporal_contrastive.py",
    ROOT / "tests" / "test_temporal_dual_fold_output_verifier.py",
    ROOT / "tests" / "test_temporal_pair_fusion.py",
    ROOT / "tests" / "test_temporal_patch_blend_kernel_builder.py",
    ROOT / "tests" / "test_temporal_patch_kernel_builder.py",
    ROOT / "tests" / "test_temporal_patch_runtime_builder.py",
    ROOT / "tests" / "test_temporal_runtime_verifier.py",
    ROOT / "tests" / "test_verify_zebrahub_contextual_dataset.py",
    ROOT / "tests" / "test_zebrahub_contextual_dataset_builder.py",
    ROOT / "tests" / "test_zebrahub_contextual_pretrain.py",
    ROOT / "tests" / "test_zebrahub_contextual_pretrain_kernel_builder.py",
    ROOT / "tests" / "test_zebrahub_external.py",
)

EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "85cfd63f75340502e3c810d71a8006fd15342dbc263f6ae45b0c376cf9b1ff7b"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "73baee5dd1ac91ab214c173c38fa86d4e9c4e508606c02c08daf764fca84e67c"
)
EXPECTED_METADATA_SHA256 = (
    "0abddc3553ac9954f9af7ee9022d80bebbc2af65d824c33d9d14ea2b78e7ae01"
)
EXPECTED_TRAIN_CACHE_BYTES = 627_198_076
EXPECTED_VALIDATION_CACHE_BYTES = 157_311_608
EXPECTED_SELECTION_TIMEPOINTS = {96, 97, 98, 99, 376, 377, 378, 379}
EXPECTED_AUDIT_TIMEPOINTS = {236, 237, 238, 239, 516, 517, 518, 519}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def check(name: str, evidence: list[Path], detail: str) -> PreflightCheck:
    return PreflightCheck.create(
        name,
        "passed",
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    return code


def loaded_tensor_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        shard = load_shard(path, torch.device("cpu"))
        total += sum(
            int(tensor.numel() * tensor.element_size()) for tensor in shard.values()
        )
    return total


def verify_sources() -> tuple[str, dict, dict, dict]:
    subprocess.run([sys.executable, str(KERNEL_BUILDER)], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(RUNTIME / "verify_runtime.py"), "--root", str(RUNTIME)],
        cwd=ROOT,
        check=True,
    )
    verification = subprocess.run(
        [
            sys.executable,
            str(RUNTIME / "verify_zebrahub_contextual_dataset.py"),
            "--root",
            str(DATASET),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dataset_evidence = json.loads(verification.stdout)
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

    if sha256_file(RUNTIME_MANIFEST) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("private runtime v8 manifest changed")
    if sha256_file(DATASET_MANIFEST) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("private ZebraHub data v4 manifest changed")
    if sha256_file(NOTEBOOK) != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("staged pretraining notebook changed")
    if sha256_file(METADATA) != EXPECTED_METADATA_SHA256:
        raise RuntimeError("staged pretraining metadata changed")

    runtime_manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    code = notebook_code()
    if len(runtime_manifest.get("files", {})) != 36:
        raise RuntimeError("private runtime v8 file inventory changed")
    if not (
        dataset_evidence.get("status") == "verified"
        and dataset_evidence.get("manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256
        and dataset_evidence.get("training", {}).get("shards") == 64
        and dataset_evidence.get("validation", {}).get("shards") == 16
        and dataset_evidence.get("competition_test_data_read") is False
        and dataset_evidence.get("public_competition_predictions_read") is False
        and dataset_evidence.get("leaderboard_used") is False
        and dataset_evidence.get("submission_created") is False
    ):
        raise RuntimeError("private ZebraHub data v4 verification changed")
    if not (
        metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and set(metadata.get("dataset_sources", []))
        == {
            "indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3",
            "indarkarhana/biohub-zebrahub-contextual-shards-v1",
        }
        and metadata.get("competition_sources") == []
        and metadata.get("kernel_sources") == []
        and metadata.get("model_sources") == []
    ):
        raise RuntimeError("two-GPU private Kaggle input policy changed")
    required_code = (
        "torch.cuda.device_count() != 2",
        "DECLARED_BUDGET_SECONDS = 24_000",
        '"--orchestrator-hard-stop-seconds", "22800"',
        '"--max-wall-seconds", "21600"',
        "selection_gate_passed",
        "audit_gate_passed",
        "result.get(\"both_folds_improved\") is True",
        "ZSNS005 disjoint developmental windows:",
        "one-shot audit",
    )
    missing_code = [fragment for fragment in required_code if fragment not in code]
    if missing_code:
        raise RuntimeError(
            f"pretraining notebook lost required execution/evidence gates: {missing_code}"
        )
    lowered_code = code.casefold()
    if "competitions submit" in lowered_code or "kaggle competitions" in lowered_code:
        raise RuntimeError("pretraining notebook contains a submission command")
    if not (
        config.get("public_code_copied") is False
        and config.get("public_predictions_copied") is False
        and config.get("public_leaderboard_used_for_selection") is False
        and config.get("submission_created") is False
        and config.get("required_visible_gpu_count") == 2
        and config.get("pretraining_execution", {}).get("required_visible_gpu_count") == 2
        and config.get("pretraining_execution", {}).get("runtime_dataset_version") == 8
        and config.get("external_data", {}).get("derived_dataset_version") == 4
    ):
        raise RuntimeError("experiment provenance or two-GPU contract changed")
    return code, runtime_manifest, dataset_manifest, config


def verify_model_and_data() -> tuple[dict, list, list]:
    production = ContextualPairFusionAssociationModel()
    parameter_count = sum(parameter.numel() for parameter in production.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"production parameter count changed: {parameter_count}")
    buffer = io.BytesIO()
    torch.save(production.state_dict(), buffer)
    buffer.seek(0)
    restored = ContextualPairFusionAssociationModel()
    restored.load_state_dict(
        torch.load(buffer, map_location="cpu", weights_only=True), strict=True
    )

    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    if not (
        smoke.get("model_family") == "temporal_contextual_pair_fusion_v3"
        and smoke.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and smoke.get("reciprocal_parent_loss_weight") == RECIPROCAL_PARENT_LOSS_WEIGHT
        and smoke.get("dataset_manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256
        and smoke.get("training_source", {}).get("finite_pair_logits") == 2800
        and smoke.get("training_source", {}).get("populated_gradients") == 64
        and smoke.get("training_source", {}).get("all_gradients_finite") is True
        and smoke.get("validation_source", {}).get("finite_pair_logits") == 1258
        and smoke.get("validation_source", {}).get("populated_gradients") == 64
        and smoke.get("validation_source", {}).get("all_gradients_finite") is True
        and smoke.get("scientific_promotion_evidence") is False
        and smoke.get("gpu_used") is False
        and smoke.get("submission_created") is False
    ):
        raise RuntimeError("production reciprocal smoke evidence changed")

    train_records = discover_shards(
        DATASET / "train",
        expected_source="ZSNS004",
        expected_role="external_pretraining",
    )
    validation_records = discover_shards(
        DATASET / "validation",
        expected_source="ZSNS005",
        expected_role="external_validation",
    )
    selection, audit = partition_validation_records(validation_records)
    if len(train_records) != 64 or len(validation_records) != 16:
        raise RuntimeError("ZebraHub shard inventory changed")
    if {record.csv_timepoint for record in selection} != EXPECTED_SELECTION_TIMEPOINTS:
        raise RuntimeError("ZSNS005 checkpoint-selection windows changed")
    if {record.csv_timepoint for record in audit} != EXPECTED_AUDIT_TIMEPOINTS:
        raise RuntimeError("ZSNS005 one-shot audit windows changed")
    train_bytes = loaded_tensor_bytes([record.path for record in train_records])
    validation_bytes = loaded_tensor_bytes([record.path for record in validation_records])
    if train_bytes != EXPECTED_TRAIN_CACHE_BYTES:
        raise RuntimeError(f"training tensor cache changed: {train_bytes}")
    if validation_bytes != EXPECTED_VALIDATION_CACHE_BYTES:
        raise RuntimeError(f"validation tensor cache changed: {validation_bytes}")
    return smoke, train_records, validation_records


def main() -> None:
    code, _, _, _ = verify_sources()
    _, train_records, validation_records = verify_model_and_data()
    model_source = RUNTIME / "contextual_pair_fusion.py"
    trainer_source = RUNTIME / "train_zebrahub_contextual_pretrain.py"
    verifier_source = RUNTIME / "verify_zebrahub_contextual_dataset.py"
    sharding_source = RUNTIME / "submission_sharding.py"
    checks = [
        check(
            "imports",
            [
                PREFLIGHT_BUILDER,
                KERNEL_BUILDER,
                NOTEBOOK,
                model_source,
                trainer_source,
                *FOCUSED_TESTS,
            ],
            "All notebook code compiles; the exact remote-redownloaded runtime verifies and the focused temporal/contextual/ZebraHub/appearance test suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, DATASET_MANIFEST, verifier_source],
            "Only private runtime v8 and balanced ZebraHub shard data v4 are attached; competition inputs are absent, internet and TPU are off, and the notebook fails closed unless exactly two T4 GPUs are visible.",
        ),
        check(
            "single_batch",
            [SMOKE, model_source, trainer_source],
            "The production 20,747,761-parameter v3 path scored 2,800 train and 1,258 validation candidate edges from exact v4 shards with finite logits.",
        ),
        check(
            "model_step",
            [SMOKE, model_source, trainer_source, ROOT / "tests" / "test_contextual_training.py"],
            "The exact outgoing-plus-0.35-incoming reciprocal objective completed backward on production shards with 64 populated finite gradients for both ZSNS004 and ZSNS005 smoke transitions.",
        ),
        check(
            "dense_memory",
            [NOTEBOOK, trainer_source, DATASET_MANIFEST],
            "Each worker preloads exactly 627,198,076 train plus 157,311,608 validation tensor bytes before constructing the 20.75M-parameter model; training and validation use CUDA autocast and the notebook requires a 16-GB T4 per fold.",
        ),
        check(
            "checkpoint_roundtrip",
            [PREFLIGHT_BUILDER, model_source, trainer_source, NOTEBOOK],
            "The exact 20,747,761-parameter production state dict strict-roundtrips locally; Kaggle terminal evidence must hash each saved fold and downstream loading remains strict and provenance-gated.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA, sharding_source],
            "The notebook writes only pretrained fold weights and terminal evidence below /kaggle/working; it attaches no competition source and contains no submission command.",
        ),
        check(
            "dataset_coverage",
            [DATASET_MANIFEST, DATASET / "train-inventory.json", DATASET / "validation-inventory.json", trainer_source],
            f"All {len(train_records)} ZSNS004 training shards and {len(validation_records)} ZSNS005 validation shards are hash-verified; checkpoint selection and one-shot audit use disjoint 8-shard developmental windows.",
        ),
        check(
            "non_replica_provenance",
            [CONFIG, TECHNIQUE_REPORT, model_source, trainer_source],
            "The temporal encoder, candidate-context pooling, reciprocal loss, and validation protocol are project-authored; public notebook code, predictions, leaderboard selection, and metric hacks are excluded.",
        ),
        check(
            "quota_policy",
            [COMPETITION_CONFIG, NOTEBOOK, CONFIG],
            "The registered ceiling is 6.67 GPU-hours versus a 24,000-second watchdog; launch remains forbidden while another GPU kernel is active or if projected remaining quota would fall below 8.00 hours.",
        ),
    ]
    if "torch.cuda.device_count() != 2" not in code:
        raise RuntimeError("two-GPU execution gate disappeared")
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
