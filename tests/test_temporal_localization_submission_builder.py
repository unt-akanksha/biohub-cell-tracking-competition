from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
DATASET = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-candidate-dataset.py"))
BUILDER = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-submission-candidate.py"))
FIXTURE = runpy.run_path(str(ROOT / "tests/test_temporal_localization_candidate_dataset.py"))


def runtime_fixture(tmp_path: Path) -> Path:
    results = FIXTURE["result_fixture"](tmp_path / "results")
    runtime = tmp_path / "runtime"
    DATASET["stage_dataset"](results, runtime)
    return runtime


def test_injected_runtime_and_helpers_compile() -> None:
    compile(BUILDER["MODEL_SETUP_TEMPLATE"], "model-setup", "exec")
    compile(BUILDER["LOCALIZATION_HELPERS"], "localization-helpers", "exec")
    compile(BUILDER["CANDIDATE_EVIDENCE"], "candidate-evidence", "exec")


def test_notebook_is_dual_t4_coordinate_only_and_non_submitting(tmp_path: Path) -> None:
    notebook = BUILDER["build_notebook"](runtime_fixture(tmp_path))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Exactly two T4 GPUs are required" in text
    assert "_TLC_MODELS_BY_DEVICE" in text
    assert "_apply_external_ranked_consensus" in text
    assert 'nodes_by_id[node_id]["z"] = float(coordinate[0])' in text
    assert '"localization_node_count_changes": 0' in text
    assert '"localization_edge_changes": 0' in text
    assert "localization_rounded_coordinate_changes" in text
    assert '"BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "1"' in text
    assert '"declared_budget_seconds": 42000' in text
    assert "Timer(40800, _biohub_budget_expired)" in text
    assert '"competition_submission_performed": False' in text
    assert "kaggle competitions submit" not in text
    assert '"public_predictions_copied": False' in text


def test_metadata_contract_remains_private_offline_and_competition_attached() -> None:
    metadata = json.loads(
        (ROOT / "kaggle/biohub-clean-0927-repro-v1/kernel-metadata.json").read_text()
    )
    derived = {
        **metadata,
        "id": f"indarkarhana/{BUILDER['TARGET_ID']}",
        "dataset_sources": [*metadata["dataset_sources"], BUILDER["RUNTIME_REF"]],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
    }
    assert derived["is_private"] is True
    assert derived["enable_gpu"] is True
    assert derived["enable_tpu"] is False
    assert derived["enable_internet"] is False
    assert derived["machine_shape"] == "NvidiaTeslaT4"
    assert BUILDER["RUNTIME_REF"] in derived["dataset_sources"]
