from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "build-temporal-contextual-submission-candidate-preflight.py"
)
SPEC = importlib.util.spec_from_file_location("contextual_candidate_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_final_notebook_policy_uses_cloud_dataset_and_no_upload() -> None:
    code, metadata = preflight.notebook_policy()
    assert metadata["dataset_sources"] == preflight.EXPECTED_DATASET_SOURCES
    assert metadata["kernel_sources"] == preflight.EXPECTED_KERNEL_SOURCES
    assert "torch.cuda.device_count() != 2" in code
    assert "competitions submit" not in code.casefold()


def test_manifest_verification_rejects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source = root / "model.pt"
    source.parent.mkdir()
    source.write_bytes(b"model")
    manifest = {
        "files": {
            "model.pt": {
                "bytes": source.stat().st_size,
                "sha256": preflight.sha256_file(source),
            }
        }
    }
    assert preflight.verify_manifest_files(root, manifest) == [source]
    source.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed"):
        preflight.verify_manifest_files(root, manifest)


def test_preflight_source_declares_every_guarded_launch_check() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "imports",
        "inputs",
        "single_batch",
        "model_step",
        "checkpoint_roundtrip",
        "output_location",
        "dense_memory",
        "dataset_coverage",
        "non_replica_provenance",
        "quota_policy",
    ):
        assert f'"{name}"' in source
    assert "--strict-checkpoint" in source
    assert "cloud_candidate_launcher_terminal.json" in source
    assert "ready_for_submission_upload" in source
    assert "competitions submit" not in preflight.notebook_policy()[0].casefold()
