from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "temporal_multiscale_contextual_pair_fusion_v4"
CANDIDATE_FAMILY = "trackastra_multiscale_contextual_pair_fusion_blend"


def build_and_read(builder_name: str, kernel_id: str) -> tuple[dict, str]:
    builder = ROOT / "scripts" / builder_name
    subprocess.run([sys.executable, str(builder)], cwd=ROOT, check=True)
    kernel = ROOT / "kaggle" / kernel_id
    metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="ascii"))
    notebook = json.loads((kernel / f"{kernel_id}.ipynb").read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(kernel / f"{kernel_id}.ipynb"), "exec")
    return metadata, code


def assert_two_gpu_private_kernel(metadata: dict, code: str) -> None:
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "torch.cuda.device_count() != 2" in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()
    assert 'public_leaderboard_used_for_selection") is False' in code


def test_multiscale_calibration_kernel_is_family_bound_and_submission_free() -> None:
    metadata, code = build_and_read(
        "build-temporal-multiscale-contextual-calibration-kernel.py",
        "biohub-multiscale-calibration-v4",
    )
    assert_two_gpu_private_kernel(metadata, code)
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-temporal-multiscale-transfer-v4",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    ]
    assert FAMILY in code
    assert "temporal-multiscale-contextual-pair-fusion-blend-v4" in code
    assert "temporal_contextual_pair_fusion_v3" not in code


def test_multiscale_processed_kernel_is_exact_gate_input_only() -> None:
    metadata, code = build_and_read(
        "build-temporal-multiscale-contextual-processed-acceptance-kernel.py",
        "biohub-multiscale-processed-v4",
    )
    assert_two_gpu_private_kernel(metadata, code)
    assert "temporal-multiscale-contextual-pair-fusion-processed-acceptance-v4" in code
    assert CANDIDATE_FAMILY in code
    assert FAMILY in code
    assert 'result.get("exact_processed_scoring_performed") is False' in code
    assert 'result.get("authorized_for_submission") is False' in code


def test_multiscale_final_runtime_is_hash_bound_and_transition_balanced() -> None:
    builder = ROOT / "scripts" / "build-temporal-multiscale-contextual-final-runtime.py"
    subprocess.run([sys.executable, str(builder), "--replace"], cwd=ROOT, check=True)
    target = (
        ROOT
        / ".biohub"
        / "staging"
        / "biohub-temporal-multiscale-contextual-final-runtime-v4"
    )
    manifest = json.loads((target / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    metadata = json.loads((target / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "indarkarhana/biohub-multiscale-contextual-final-v4"
    assert manifest["parent_runtime"]["dataset_version"] == 3
    assert manifest["integrity"]["final_appearance_family"] == FAMILY
    assert manifest["integrity"]["parameters_per_fold"] == 46_386_607
    assert manifest["integrity"]["transition_partitioned_inference"] is True
    assert manifest["integrity"]["each_consecutive_transition_processed_exactly_once"] is True
    for name, row in manifest["files"].items():
        path = target / name
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    subprocess.run(
        [sys.executable, str(target / "verify_runtime.py"), "--root", str(target)],
        cwd=ROOT,
        check=True,
    )


def test_multiscale_candidate_requires_exact_acceptance_and_is_non_replica() -> None:
    metadata, code = build_and_read(
        "build-temporal-multiscale-contextual-submission-kernel.py",
        "biohub-multiscale-submission-candidate-v4",
    )
    assert_two_gpu_private_kernel(metadata, code)
    assert metadata["dataset_sources"][:2] == [
        "indarkarhana/biohub-multiscale-contextual-final-v4",
        "indarkarhana/biohub-multiscale-exact-acceptance-v4",
    ]
    assert FAMILY in code
    assert CANDIDATE_FAMILY in code
    assert 'payload.get("exact_processed_gate_passed") is True' in code
    assert 'report.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256' in code
    assert 'report.get("transition_partitioned_inference") is True' in code
    assert 'shutil.copy2(candidate, "/kaggle/working/submission.csv")' in code


def test_multiscale_exact_stage_and_preflight_transforms_are_fail_closed() -> None:
    wrappers = {
        "run-temporal-multiscale-contextual-exact-acceptance.py": (
            "temporal-multiscale-contextual-pair-fusion-processed-acceptance-v4",
            FAMILY,
            CANDIDATE_FAMILY,
        ),
        "stage-temporal-multiscale-kaggle-artifacts.py": (
            "indarkarhana/biohub-multiscale-exact-acceptance-v4",
            FAMILY,
            CANDIDATE_FAMILY,
        ),
        "build-temporal-multiscale-submission-preflight.py": (
            "biohub-multiscale-submission-candidate-v4",
            FAMILY,
            CANDIDATE_FAMILY,
        ),
    }
    for name, required in wrappers.items():
        namespace = runpy.run_path(str(ROOT / "scripts" / name))
        transformed = namespace["transformed_source"]()
        compile(transformed, name, "exec")
        assert all(fragment in transformed for fragment in required)
        if "preflight" not in name:
            assert "competitions submit" not in transformed.casefold()
            assert "kaggle competitions" not in transformed.casefold()
        else:
            assert 'if "competitions submit" in code.casefold()' in transformed
        assert "public_leaderboard_used_for_selection" in transformed


def test_multiscale_background_chain_transforms_validate_without_side_effects() -> None:
    stages = {
        "wait-verify-launch-temporal-multiscale-calibration.ps1": (
            "multiscale_calibration"
        ),
        "wait-verify-launch-temporal-multiscale-processed.ps1": (
            "multiscale_processed"
        ),
        "wait-score-stage-launch-temporal-multiscale-final.ps1": "multiscale_final",
        "wait-verify-submit-temporal-multiscale-final.ps1": "multiscale_submit",
    }
    for name, stage in stages.items():
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / name),
                "-ValidateOnly",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["status"] == "validated"
        assert payload["stage"] == stage
        assert payload["transformed_characters"] > 5_000
