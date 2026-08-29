from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_BUILDER = ROOT / "scripts/build-learned-division-recovery-runtime.py"
SUBMISSION_BUILDER = ROOT / "scripts/build-learned-division-submission-candidate.py"


def make_runtime(tmp_path: Path) -> Path:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted",
                "run_id": "external-division-recovery-policy-v1",
                "model_training_run_id": "focused-division-gate-v1",
                "appearance_family": "temporal_multiscale_contextual_pair_fusion_v4",
                "focused_division_family": "temporal_multiscale_focused_division_gate_v1",
                "model_sha256": {
                    "target_44b6": "a" * 64,
                    "target_6bba": "b" * 64,
                },
                "frozen_division_logit_threshold": 1.25,
                "audit_opened_after_threshold_freeze": True,
                "competition_data_read": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "authorized_for_competition_graph_evaluation": True,
                "authorized_for_submission": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runtime"
    module = runpy.run_path(str(RUNTIME_BUILDER))
    previous = sys.argv
    try:
        sys.argv = [
            str(RUNTIME_BUILDER),
            "--policy",
            str(policy),
            "--output-root",
            str(output),
        ]
        module["main"]()
    finally:
        sys.argv = previous
    return output


def test_candidate_is_attributed_two_gpu_and_uses_learned_division_gate(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    module = runpy.run_path(str(SUBMISSION_BUILDER))
    assert module["sha256_file"](module["SOURCE_NOTEBOOK"]) == module[
        "SOURCE_NOTEBOOK_SHA256"
    ]
    notebook = module["transform_notebook"](runtime)
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "grafael/biohub-ct-0940-ema" in source
    assert 'BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"' in source
    assert "velocity_ema_alpha = 0.4" in source
    assert "external-division-recovery-policy-v1" in source
    assert "_apply_external_learned_division_recovery" in source
    assert "Exactly two T4 GPUs are required" in source
    assert "kaggle competitions submit" not in source
    assert "candidate_evidence.json" in source
    assert '"target_public_score": 0.945' in source
    assert source.index("def _apply_external_learned_division_recovery") < source.index(
        "def filter_output_graph("
    )
    assert source.index("edges, learned_division_stats") < source.index(
        "return nodes_by_id, edges, stats"
    )
    assert source.index("candidate_evidence.json") < source.index(
        '_biohub_write_terminal("completed")'
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_candidate_metadata_attaches_private_runtime_and_focused_gate(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    module = runpy.run_path(str(SUBMISSION_BUILDER))
    previous = sys.argv
    target = tmp_path / "candidate"
    target_notebook = target / "candidate.ipynb"
    main_globals = module["main"].__globals__
    main_globals["TARGET_DIR"] = target
    main_globals["TARGET_NOTEBOOK"] = target_notebook
    try:
        sys.argv = [
            str(SUBMISSION_BUILDER),
            "--runtime-root",
            str(runtime),
        ]
        module["main"]()
    finally:
        sys.argv = previous
    assert target_notebook.is_file()
    metadata = json.loads((target / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"][-2:] == [
        "indarkarhana/biohub-learned-division-recovery-runtime-v1",
        "indarkarhana/biohub-focused-division-gate-v1",
    ]
    assert metadata["kernel_sources"] == []


def test_main_reuses_only_an_empty_generated_target(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    module = runpy.run_path(str(SUBMISSION_BUILDER))
    target = tmp_path / "candidate"
    target.mkdir()
    target_notebook = target / "candidate.ipynb"
    main_globals = module["main"].__globals__
    main_globals["TARGET_DIR"] = target
    main_globals["TARGET_NOTEBOOK"] = target_notebook
    previous = sys.argv
    try:
        sys.argv = [str(SUBMISSION_BUILDER), "--runtime-root", str(runtime)]
        module["main"]()
    finally:
        sys.argv = previous

    assert target_notebook.is_file()
