#!/usr/bin/env python
"""Build the gated graph-context plus temporal-localization composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py"))
GRAPH = runpy.run_path(str(ROOT / "scripts/build-graph-context-consensus-submission-candidate.py"))
LOCALIZATION = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-submission-candidate.py"))
GRAPH_DATASET = runpy.run_path(str(ROOT / "scripts/build-graph-context-consensus-division-dataset.py"))
LOCALIZATION_DATASET = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-candidate-dataset.py"))

SOURCE_DIR = COMMON["SOURCE_DIR"]
SOURCE_METADATA_SHA256 = COMMON["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-ema-graph-localization-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
GRAPH_RUNTIME_REF = GRAPH["CONSENSUS_REF"]
LOCALIZATION_RUNTIME_REF = LOCALIZATION["RUNTIME_REF"]
RUN_ID = "ema-graph-localization-composition-v1"


ATTRIBUTION = """## Project candidate: independently gated graph and localization composition

The detector, harmonic association control, gap handling, safe-division rule,
and graph reconstruction retain their original public attribution. No public
prediction is copied. Both additive components are project-authored.

The graph-context component may add at most one parent-free division edge per
movie using an equal-rank ensemble and an independent morphology voter. The
temporal-localization component then changes only coordinates, with
multi-member direction, uncertainty, safe-radius, disagreement, boundary, and
global move-fraction gates. Every attached checkpoint passed its own frozen
selection and sealed audit before either component was evaluated on the exact
candidate. This composition is eligible for submission only if both standalone
components independently pass their external promotion gates and the composed
output improves on both standalone candidates. No leaderboard result selects
models, weights, thresholds, edges, coordinates, or the decision to compose.
Exactly two T4 GPUs are required; both model families are partitioned across
the same two devices and run sequentially, graph first and localization second.
"""


def _rename_apply(source: str, replacement: str) -> str:
    old = "def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):"
    if source.count(old) != 1:
        raise RuntimeError("component apply contract changed")
    return source.replace(old, replacement, 1)


GRAPH_HELPERS = _rename_apply(
    GRAPH["RANKED_HELPERS"],
    "def _apply_graph_context_consensus(nodes_by_id, edges, dataset):",
)
LOCALIZATION_HELPERS = _rename_apply(
    LOCALIZATION["LOCALIZATION_HELPERS"],
    "def _apply_temporal_localization(nodes_by_id, edges, dataset):",
)

COMPOSITION_WRAPPER = r'''
def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    edges, graph_stats = _apply_graph_context_consensus(nodes_by_id, edges, dataset)
    edges, localization_stats = _apply_temporal_localization(nodes_by_id, edges, dataset)
    localization_only = {
        key: value
        for key, value in localization_stats.items()
        if key.startswith("localization_")
    }
    required = {
        "localization_candidate_nodes",
        "localization_nodes_moved",
        "localization_rounded_coordinate_changes",
        "localization_global_gate_failures",
        "localization_node_count_changes",
        "localization_edge_changes",
        "localization_gpu_groups_used",
    }
    if not required <= set(localization_only):
        raise RuntimeError("Temporal-localization composition stats are incomplete")
    return edges, {**graph_stats, **localization_only}

'''

COMPOSITION_HELPERS = GRAPH_HELPERS + LOCALIZATION_HELPERS + COMPOSITION_WRAPPER


CANDIDATE_EVIDENCE = r'''# Emit hash-bound evidence for the external composition gate.
_evidence_submission = Path("/kaggle/working/submission.csv")
_evidence_stats = pd.read_csv(RUN_STATS_PATH)
_evidence_validator = pd.read_csv(VALIDATOR_STATS_PATH)
_evidence_weight = float(_evidence_validator["weight"].sum())
if _evidence_weight <= 0:
    raise RuntimeError("Candidate validator emitted no weighted rows")
_evidence_adjusted_edge = float(
    (_evidence_validator["adjusted_edge_jaccard"] * _evidence_validator["weight"]).sum()
    / _evidence_weight
)
_evidence_div_tp = int(_evidence_validator["div_tp"].sum())
_evidence_div_fp = int(_evidence_validator["div_fp"].sum())
_evidence_div_fn = int(_evidence_validator["div_fn"].sum())
_evidence_div_denominator = _evidence_div_tp + _evidence_div_fp + _evidence_div_fn
_evidence_division = (
    _evidence_div_tp / _evidence_div_denominator
    if _evidence_div_denominator
    else 0.0
)
_evidence = {
    "schema_version": 1,
    "status": "completed_pending_external_promotion_gate",
    "run_id": "ema-graph-localization-composition-v1",
    "target_public_score": 0.945,
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "component_order": ["graph_context_division", "temporal_localization"],
    "independent_component_promotion_required": True,
    "graph_runtime_manifest_sha256": _GCD_MANIFEST_SHA256,
    "graph_policy": _GCD_POLICY["graph_context_policy"],
    "graph_member_count": len(_GCD_DEEP_MODELS),
    "graph_parameters_per_member": 74_732_308,
    "graph_model_sha256": [row["model_sha256"] for row in _GCD_POLICY["graph_context_members"]],
    "morphology_model_sha256": _GCD_POLICY["morphology_model_sha256"],
    "localization_runtime_manifest_sha256": _TLC_MANIFEST_SHA256,
    "localization_member_count": len(_TLC_MODELS),
    "localization_parameters_per_member": 71_249_805,
    "localization_model_sha256": [row["model_sha256"] for row in _TLC_MEMBERS],
    "absolute_threshold_used": False,
    "node_count_preserving": True,
    "model_subset_searched_on_audit": False,
    "weights_searched_on_development": False,
    "threshold_searched_on_development": False,
    "ranked_geometric_candidates": int(_evidence_stats["ranked_consensus_geometric_candidates"].sum()),
    "ranked_geometry_eligible_candidates": int(_evidence_stats["ranked_consensus_geometry_eligible_candidates"].sum()),
    "ranked_candidate_parents_scored": int(_evidence_stats["ranked_consensus_candidate_parents_scored"].sum()),
    "ranked_agreements": int(_evidence_stats["ranked_consensus_ranking_agreed"].sum()),
    "ranked_edges_added": int(_evidence_stats["ranked_consensus_added_edges"].sum()),
    "ranked_reassignments": int(_evidence_stats["ranked_consensus_reassignment_performed"].sum()),
    "ranked_node_or_coordinate_changes": int(_evidence_stats["ranked_consensus_node_or_coordinate_changes"].sum()),
    "graph_gpu_groups_used": int(_evidence_stats["ranked_consensus_gpu_groups_used"].sum()),
    "localization_candidate_nodes": int(_evidence_stats["ranked_consensus_localization_candidate_nodes"].sum()),
    "localization_nodes_moved": int(_evidence_stats["ranked_consensus_localization_nodes_moved"].sum()),
    "localization_boundary_rejected": int(_evidence_stats["ranked_consensus_localization_boundary_rejected"].sum()),
    "localization_rounded_coordinate_changes": int(_evidence_stats["ranked_consensus_localization_rounded_coordinate_changes"].sum()),
    "localization_global_gate_failures": int(_evidence_stats["ranked_consensus_localization_global_gate_failures"].sum()),
    "localization_node_count_changes": int(_evidence_stats["ranked_consensus_localization_node_count_changes"].sum()),
    "localization_edge_changes": int(_evidence_stats["ranked_consensus_localization_edge_changes"].sum()),
    "localization_gpu_groups_used": int(_evidence_stats["ranked_consensus_localization_gpu_groups_used"].sum()),
    "validator_adjusted_edge_jaccard": _evidence_adjusted_edge,
    "validator_division_tp": _evidence_div_tp,
    "validator_division_fp": _evidence_div_fp,
    "validator_division_fn": _evidence_div_fn,
    "validator_division_jaccard": _evidence_division,
    "validator_proxy_score": _evidence_adjusted_edge + 0.10 * _evidence_division,
    "validator_missed_gt_nodes": int(_evidence_validator["missed_gt_nodes"].sum()),
    "validator_spurious_pred_nodes": int(_evidence_validator["spurious_pred_nodes"].sum()),
    "submission_sha256": _tlc_sha256(_evidence_submission),
    "competition_submission_performed": False,
    "authorized_for_submission": False,
}
_evidence_path = Path("/kaggle/working/candidate_evidence.json")
_evidence_path.write_text(
    _tlc_json.dumps(_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_tlc_json.dumps(_evidence, indent=2, sort_keys=True))
'''


def _extend_watchdog(notebook: dict) -> None:
    watchdog = "".join(notebook["cells"][0]["source"])
    replacements = {
        "# Biohub quota watchdog: 3600 s declared budget, 600 s safety margin.":
            "# Biohub quota watchdog: 43200 s declared budget, 1200 s safety margin.",
        '"declared_budget_seconds": 3600,': '"declared_budget_seconds": 43200,',
        '"safety_margin_seconds": 600,': '"safety_margin_seconds": 1200,',
        "_BIOHUB_TIMER = _biohub_threading.Timer(3000, _biohub_budget_expired)":
            "_BIOHUB_TIMER = _biohub_threading.Timer(42000, _biohub_budget_expired)",
        'print("Biohub watchdog armed: hard stop after 3000 seconds.")':
            'print("Biohub watchdog armed: hard stop after 42000 seconds.")',
    }
    for old, new in replacements.items():
        if watchdog.count(old) != 1:
            raise RuntimeError(f"Attributed watchdog contract changed: {old}")
        watchdog = watchdog.replace(old, new)
    notebook["cells"][0]["source"] = watchdog.splitlines(keepends=True)


def build_notebook(graph_root: Path, localization_root: Path) -> dict:
    graph_verified = GRAPH_DATASET["verify_dataset"](graph_root)
    localization_verified = LOCALIZATION_DATASET["verify_dataset"](localization_root)
    model_setup = (
        GRAPH["MODEL_SETUP_TEMPLATE"].replace(
            "__MANIFEST_SHA256__", graph_verified["manifest_sha256"]
        )
        + "\n"
        + LOCALIZATION["MODEL_SETUP_TEMPLATE"].replace(
            "__MANIFEST_SHA256__", localization_verified["manifest_sha256"]
        )
    )
    notebook = COMMON["transform_notebook"](
        graph_root,
        dataset_builder_path=GRAPH["DATASET_BUILDER"],
        watchdog_run_id=RUN_ID,
        keep_base_safe_divisions=True,
        attribution=ATTRIBUTION,
        model_setup_template=model_setup,
        ranked_helpers=COMPOSITION_HELPERS,
        candidate_evidence=CANDIDATE_EVIDENCE,
    )
    _extend_watchdog(notebook)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--localization-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source_metadata_path = SOURCE_DIR / "kernel-metadata.json"
    if COMMON["sha256_file"](source_metadata_path) != SOURCE_METADATA_SHA256:
        raise RuntimeError("Attributed public-control metadata changed")
    if TARGET_DIR.exists():
        if not any(TARGET_DIR.iterdir()):
            TARGET_DIR.rmdir()
        elif not args.replace:
            raise FileExistsError(TARGET_DIR)
        else:
            shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook(args.graph_root, args.localization_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub EMA Graph plus Localization v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "ensemble", "non-replica"],
        "dataset_sources": [
            *base_metadata["dataset_sources"],
            GRAPH_RUNTIME_REF,
            LOCALIZATION_RUNTIME_REF,
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
