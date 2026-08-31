from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-localization-composition-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))


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
