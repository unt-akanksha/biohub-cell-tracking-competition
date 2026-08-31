from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-localization-composition-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))
TEMPORAL_FIXTURE = runpy.run_path(
    str(ROOT / "tests/test_temporal_localization_candidate_dataset.py")
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _graph_runtime(root: Path) -> Path:
    root.mkdir(parents=True)
    dataset = MODULE["GRAPH_DATASET"]
    model_name = "graph_context_division_model_00.pt"
    required = {
        *dataset["RUNTIME_FILES"],
        model_name,
        "selection_audit_terminal.json",
        "graph_context_development_probe.json",
        "graph_context_development_evidence.json",
        "morphology_division_model.joblib",
        "morphology_training_terminal.json",
        "graph-context-consensus-policy.json",
        dataset["SKLEARN_WHEEL_NAME"],
    }
    for name in required - {"graph-context-consensus-policy.json"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture-" + name).encode())
    sha256_file = dataset["sha256_file"]
    model_hash = sha256_file(root / model_name)
    morphology_hash = sha256_file(root / "morphology_division_model.joblib")
    wheel_hash = sha256_file(root / dataset["SKLEARN_WHEEL_NAME"])
    _write_json(
        root / "graph-context-consensus-policy.json",
        {
            "schema_version": 1,
            "status": "development_accepted",
            "run_id": dataset["POLICY_RUN_ID"],
            "graph_context_policy": "strongest_selection_individual",
            "graph_context_member_count": 1,
            "graph_context_members": [
                {
                    "path": model_name,
                    "model_sha256": model_hash,
                    "parameter_count": dataset["EXPECTED_PARAMETER_COUNT"],
                    "selection_gate_passed": True,
                    "audit_gate_passed": True,
                }
            ],
            "morphology_model_sha256": morphology_hash,
            "sklearn_version": dataset["SKLEARN_VERSION"],
            "sklearn_wheel_sha256": wheel_hash,
            "biological_geometry_minimum": 3.0,
            "maximum_added_edges_per_movie": 1,
            "base_safe_division_heuristic_enabled": True,
            "external_policy_additive_only": True,
            "exact_two_t4_required": True,
            "absolute_threshold_used": False,
            "model_subset_searched_on_audit": False,
            "authorized_for_full_candidate_evaluation": True,
            "authorized_for_submission": False,
        },
    )
    _write_json(
        root / "GRAPH_CONTEXT_CONSENSUS_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": dataset["RUN_ID"],
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "files": {
                name: {"path": name, "sha256": sha256_file(root / name)}
                for name in required
            },
        },
    )
    return root


def _localization_runtime(root: Path) -> Path:
    results = TEMPORAL_FIXTURE["result_fixture"](root.parent / "localization-results")
    MODULE["LOCALIZATION_DATASET"]["stage_dataset"](results, root)
    return root


def test_composition_code_compiles_and_orders_components() -> None:
    compile(MODULE["GRAPH"]["MODEL_SETUP_TEMPLATE"], "graph-setup", "exec")
    compile(MODULE["LOCALIZATION"]["MODEL_SETUP_TEMPLATE"], "localizer-setup", "exec")
    compile(MODULE["COMPOSITION_HELPERS"], "composition-helpers", "exec")
    compile(MODULE["CANDIDATE_EVIDENCE"], "composition-evidence", "exec")
    helpers = MODULE["COMPOSITION_HELPERS"]
    graph_call = helpers.index("_apply_graph_context_consensus(nodes_by_id, edges, dataset)")
    localization_call = helpers.index("_apply_temporal_localization(nodes_by_id, edges, dataset)")
    assert graph_call < localization_call
    assert helpers.count("def _apply_external_ranked_consensus(") == 1


def test_composition_is_dual_gpu_non_replica_and_non_submitting() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    setup = (
        MODULE["GRAPH"]["MODEL_SETUP_TEMPLATE"]
        + MODULE["LOCALIZATION"]["MODEL_SETUP_TEMPLATE"]
    )
    assert "Exactly two T4 GPUs are required" in setup
    assert "_GCD_MODELS_BY_DEVICE" in setup
    assert "_TLC_MODELS_BY_DEVICE" in setup
    assert "No public" in MODULE["ATTRIBUTION"]
    assert "prediction is copied" in MODULE["ATTRIBUTION"]
    assert "independently pass their external promotion gates" in MODULE["ATTRIBUTION"]
    assert '"independent_component_promotion_required": True' in MODULE["CANDIDATE_EVIDENCE"]
    assert '"component_order": ["graph_context_division", "temporal_localization"]' in MODULE["CANDIDATE_EVIDENCE"]
    assert "kaggle competitions submit" not in source
    assert '"competition_submission_performed": False' in source


def test_metadata_contract_attaches_both_private_runtimes() -> None:
    assert MODULE["SOURCE_DIR"] == MODULE["COMMON"]["SOURCE_DIR"]
    assert MODULE["GRAPH_RUNTIME_REF"] != MODULE["LOCALIZATION_RUNTIME_REF"]
    assert MODULE["GRAPH_RUNTIME_REF"].startswith("indarkarhana/")
    assert MODULE["LOCALIZATION_RUNTIME_REF"].startswith("indarkarhana/")
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"is_private": True' in source
    assert '"enable_internet": False' in source
    assert '"machine_shape": "NvidiaTeslaT4"' in source


def test_full_notebook_transform_binds_both_runtime_manifests(tmp_path: Path) -> None:
    graph = _graph_runtime(tmp_path / "graph-runtime")
    localization = _localization_runtime(tmp_path / "localization-runtime")
    notebook = MODULE["build_notebook"](graph, localization)
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    graph_hash = MODULE["GRAPH_DATASET"]["verify_dataset"](graph)["manifest_sha256"]
    localization_hash = MODULE["LOCALIZATION_DATASET"]["verify_dataset"](
        localization
    )["manifest_sha256"]
    assert graph_hash in text
    assert localization_hash in text
    assert "__MANIFEST_SHA256__" not in text
    assert text.index("_apply_graph_context_consensus") < text.index(
        "_apply_temporal_localization"
    )
    assert '"declared_budget_seconds": 43200' in text
    assert "Timer(42000, _biohub_budget_expired)" in text
    assert "kaggle competitions submit" not in text
