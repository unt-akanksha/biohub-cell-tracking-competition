from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT / "scripts" / "build-temporal-contextual-submission-candidate-kernel.py"
)
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-contextual-submission-candidate-v3"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-contextual-submission-candidate-v3.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"


def build() -> tuple[dict, dict, str]:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, metadata, code


def test_candidate_kernel_is_accepted_two_gpu_whole_movie_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-contextual-submission-candidate-v3"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
        "indarkarhana/biohub-temporal-contextual-transfer-output-v3",
        "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-clean-0-927-reproduction-v1",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 43_200" in code
    assert "INFERENCE_HARD_STOP_SECONDS = 36_000" in code
    assert "FINALIZATION_RESERVE_SECONDS = 7_200" in code
    assert '"--hard-stop-seconds", "36000"' in code
    assert 'payload.get("status") == "accepted"' in code
    assert 'payload.get("exact_processed_gate_passed") is True' in code
    assert '"trackastra_contextual_pair_fusion_blend"' in code
    assert '"temporal_contextual_pair_fusion_v3"' in code
    assert '"--strict-checkpoint"' in code
    assert (
        "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
        in code
    )
    assert 'report.get("gpu_count") == 2' in code
    assert 'report.get("whole_movie_coverage", [])' in code
    assert 'report.get("nodes_preserved_exactly") is True' in code
    assert 'report.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256' in code
    assert 'int(report.get("total_changed_edges", 0)) > 0' in code
    assert 'report.get("competition_submission_performed") is False' in code
    assert 'shutil.copy2(candidate, "/kaggle/working/submission.csv")' in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_candidate_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
