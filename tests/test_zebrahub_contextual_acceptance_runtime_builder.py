from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-acceptance-runtime.py"
TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-zebrahub-contextual-acceptance-runtime-v1"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    subprocess.run([sys.executable, str(BUILDER), "--replace"], check=True)
    return json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))


def test_acceptance_runtime_is_minimal_hash_bound_and_two_gpu() -> None:
    manifest = build()
    metadata = json.loads(
        (TARGET / "dataset-metadata.json").read_text(encoding="utf-8")
    )

    assert manifest["run_id"] == "zebrahub-contextual-acceptance-runtime-v1"
    assert metadata["id"] == (
        "indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1"
    )
    assert metadata["isPrivate"] is True
    assert manifest["integrity"]["required_gpu_count"] == 2
    assert manifest["integrity"]["one_shot_acceptance_hard_stop_seconds"] == 3_600
    assert manifest["integrity"]["competition_data_read"] is False
    assert manifest["integrity"]["competition_submission_command_included"] is False
    assert set(manifest["files"]) == {
        "contextual_pair_fusion.py",
        "evaluate_zebrahub_contextual_acceptance.py",
        "hybrid_linker.py",
        "model.py",
        "pair_fusion.py",
        "patch_model.py",
        "submission_sharding.py",
        "synthetic_data.py",
        "train_dual_fold_patch.py",
        "train_zebrahub_contextual_pretrain.py",
        "trainer.py",
        "transition_context.py",
        "verify_runtime.py",
        "verify_zebrahub_contextual_acceptance.py",
        "verify_zebrahub_contextual_dataset.py",
    }
    for relative, expected in manifest["files"].items():
        path = TARGET / relative
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    evaluator = (TARGET / "evaluate_zebrahub_contextual_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert "torch.cuda.device_count() != 2" in evaluator
    assert "kaggle competitions submit" not in evaluator.casefold()


def test_acceptance_runtime_is_byte_deterministic() -> None:
    build()
    first = {
        path.name: path.read_bytes()
        for path in TARGET.iterdir()
        if path.is_file()
    }

    build()

    assert {
        path.name: path.read_bytes()
        for path in TARGET.iterdir()
        if path.is_file()
    } == first


def test_acceptance_runtime_passes_its_portable_verifier() -> None:
    build()

    completed = subprocess.run(
        [
            sys.executable,
            str(TARGET / "verify_runtime.py"),
            "--root",
            str(TARGET),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(completed.stdout)

    assert evidence["status"] == "verified"
    assert evidence["files_checked"] == 15
    assert evidence["required_gpu_count"] == 2
    assert evidence["submission_command_included"] is False
