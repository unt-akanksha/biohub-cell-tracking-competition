#!/usr/bin/env python
"""Build the additive EMA plus audit-passing graph-context candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py"))
STRONG = runpy.run_path(str(ROOT / "scripts/build-strong-member-consensus-submission-candidate.py"))
RELATIONAL = runpy.run_path(str(ROOT / "scripts/build-relational-consensus-submission-candidate.py"))
DATASET_BUILDER = ROOT / "scripts/build-graph-context-consensus-division-dataset.py"
SOURCE_DIR = COMMON["SOURCE_DIR"]
SOURCE_METADATA_SHA256 = COMMON["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-ema-graph-context-consensus-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
CONSENSUS_REF = "indarkarhana/biohub-graph-context-consensus-division-v1"
RUN_ID = "ema-graph-context-consensus-candidate-v1"


ATTRIBUTION = """## Project candidate: EMA control plus heavy graph-context consensus

The detection, harmonic association control, safe-division rule, and graph
reconstruction retain their original public attribution. No public prediction
is copied. The additive division stage is project-authored and evaluates the
exact parent/retained-daughter/proposed-daughter tuple used during official-
train simulation.

Only 74.7M-parameter checkpoints that passed movie-disjoint selection are
attached. Their immutable deployment policy passed sealed audit either as a
precommitted ensemble unit or, under the original contract, independently.
Ensemble membership was frozen before audit. Each model combines three local image volumes with a
daughter-symmetric transformer over the surrounding detection set. Within-
movie ranks are averaged without an absolute threshold, and a proposed edge
must be the identical top geometry-eligible candidate under the graph-context
voter and an independent 132-feature temporal morphology voter. At most one
parent-free edge is added per movie. The notebook requires two T4 GPUs and
partitions accepted graph-context members across them when at least two
independently strong members were admitted.
"""


def _graph_model_setup() -> str:
    """Mechanically preserve the audited loader while changing model family."""
    source = RELATIONAL["MODEL_SETUP_TEMPLATE"]
    source = source.replace("_RCD", "_GCD").replace("_rcd", "_gcd")
    replacements = (
        ("additive relational-consensus runtime", "additive graph-context-consensus runtime"),
        ("RELATIONAL_CONSENSUS_MANIFEST.json", "GRAPH_CONTEXT_CONSENSUS_MANIFEST.json"),
        (
            "competition-relational-consensus-division-dataset-v1",
            "competition-graph-context-consensus-division-dataset-v1",
        ),
        ("relational-consensus-policy.json", "graph-context-consensus-policy.json"),
        ("relational_development_probe.json", "graph_context_development_probe.json"),
        ("relational_development_evidence.json", "graph_context_development_evidence.json"),
        (
            "competition-relational-consensus-division-policy-v1",
            "competition-graph-context-consensus-division-policy-v1",
        ),
        ("relational_members", "graph_context_members"),
        ("relational_member_count", "graph_context_member_count"),
        ("relational_policy", "graph_context_policy"),
        ("48_313_050", "74_732_308"),
        ("Relational source evidence", "Graph-context source evidence"),
        ("Relational runtime changed", "Graph-context runtime changed"),
        ("Relational parameter inventory", "Graph-context parameter inventory"),
        ("Relational morphology payload", "Graph-context morphology payload"),
        ("relational-sklearn-1.9.0", "graph-context-sklearn-1.9.0"),
        ('"relational_root"', '"graph_context_root"'),
        ('"relational_member_count"', '"graph_context_member_count"'),
        ('"relational_policy"', '"graph_context_policy"'),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    old_audit_gate = '        and row.get("audit_gate_passed") is True'
    new_audit_gate = '''        and (
            row.get("audit_gate_passed") is True
            or (
                _GCD_POLICY.get("constituent_audit_gate_required") is False
                and _GCD_POLICY.get("policy_unit_audited") is True
                and _GCD_POLICY.get("policy_contract")
                == "all-selection-admitted-equal-rank-ensemble-v2"
            )
        )'''
    if old_audit_gate not in source:
        raise RuntimeError("relational member audit contract changed")
    source = source.replace(old_audit_gate, new_audit_gate)
    old_imports = """from relational_division_inference import score_relational_candidates as _gcd_score_relational_candidates
from relational_division_model import RelationalDivisionModel as _GCDDeepModel"""
    new_imports = """from graph_context_division_inference import member_scores as _gcd_member_scores
from graph_context_division_model import GraphContextDivisionModel as _GCDDeepModel
from graph_context_features import context_tokens as _gcd_context_tokens, physical_nodes as _gcd_physical_nodes
from relational_division_inference import (
    calibration_free_parent_scores as _gcd_calibration_free_parent_scores,
    candidate_centers_zyx as _gcd_candidate_centers_zyx,
    candidate_geometry_features as _gcd_candidate_geometry_features,
)"""
    if old_imports not in source:
        raise RuntimeError("relational loader import contract changed")
    source = source.replace(old_imports, new_imports)
    old_append = "_GCD_MODELS_BY_DEVICE[_device_index].append(_model)"
    if old_append not in source:
        raise RuntimeError("relational dual-GPU partition contract changed")
    source = source.replace(
        old_append,
        "_GCD_MODELS_BY_DEVICE[_device_index].append((_index, _model))",
    )
    return source


MODEL_SETUP_TEMPLATE = _graph_model_setup()


RANKED_HELPERS = r'''
def _graph_context_scores_for_candidates(nodes_by_id, edges, dataset):
    candidates = _discover_division_candidates(nodes_by_id, edges)
    eligible = [
        candidate for candidate in candidates
        if candidate.biological_geometry_score >= _GCD_GEOMETRY_MINIMUM
    ]
    if not eligible:
        return {}, {}, {
            "candidate_parents_scored": 0,
            "candidate_frames_scored": 0,
            "deep_member_count": len(_GCD_DEEP_MODELS),
            "gpu_groups_used": 0,
        }
    parent_ids = sorted(candidate.parent_id for candidate in eligible)
    if len(set(parent_ids)) != len(parent_ids):
        raise RuntimeError("Graph-context candidates must have unique parents")
    by_time = {}
    for candidate in eligible:
        by_time.setdefault(int(nodes_by_id[candidate.parent_id]["t"]), []).append(candidate)
    maximum_time = max(int(node["t"]) for node in nodes_by_id.values())
    physical = _gcd_physical_nodes(nodes_by_id)
    raw_scores_by_model = [dict() for _model in _GCD_DEEP_MODELS]
    morphology_scores = {}
    frame_cache = {}
    active_groups = [index for index, rows in enumerate(_GCD_MODELS_BY_DEVICE) if rows]

    for timepoint, frame_candidates in sorted(by_time.items()):
        frame_indices = [
            max(0, min(maximum_time, timepoint + offset))
            for offset in (-1, 0, 1)
        ]
        temporal = np.stack(
            [read_test_frame(dataset, index, frame_cache) for index in frame_indices],
            axis=0,
        )
        centers = np.concatenate(
            [_gcd_candidate_centers_zyx(candidate, nodes_by_id) for candidate in frame_candidates],
            axis=0,
        ).astype(np.float32)
        geometry = np.stack(
            [_gcd_candidate_geometry_features(candidate, nodes_by_id, edges) for candidate in frame_candidates]
        ).astype(np.float32)
        context_rows = [
            _gcd_context_tokens(
                physical,
                parent_id=candidate.parent_id,
                existing_child_id=candidate.existing_child_id,
                proposed_child_id=candidate.second_child_id,
            )
            for candidate in frame_candidates
        ]
        context = np.stack([row[0] for row in context_rows]).astype(np.float32)
        context_mask = np.stack([row[1] for row in context_rows]).astype(np.bool_)

        def _score_device_group(device_index):
            indexed_models = _GCD_MODELS_BY_DEVICE[device_index]
            device = _GCD_DEVICES[device_index]
            sampled = _gcd_sample_physical_patches(
                _gcd_torch.as_tensor(temporal, device=device),
                centers,
                voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
                chunk_size=24,
            )
            expected = (len(frame_candidates) * 3, 3, 17, 17, 17)
            if tuple(sampled.shape) != expected:
                raise RuntimeError(f"Graph-context sampled patch shape changed: {tuple(sampled.shape)}")
            patches = sampled.reshape(len(frame_candidates), 3, 3, 17, 17, 17)
            member_values = _gcd_member_scores(
                [row[1] for row in indexed_models],
                patches,
                _gcd_torch.as_tensor(geometry),
                _gcd_torch.as_tensor(context),
                _gcd_torch.as_tensor(context_mask),
                device=device,
                batch_size=8,
            )
            parent_patches = (
                patches[:, 0].float().cpu().numpy()
                if device_index == active_groups[0]
                else None
            )
            del sampled, patches
            return [row[0] for row in indexed_models], member_values, parent_patches

        with _gcd_futures.ThreadPoolExecutor(max_workers=len(active_groups)) as executor:
            group_results = list(executor.map(_score_device_group, active_groups))
        parent_patches = next(row[2] for row in group_results if row[2] is not None)
        for model_indices, member_values, _parent_patches in group_results:
            for model_index, values in zip(model_indices, member_values, strict=True):
                for candidate, value in zip(frame_candidates, values.tolist(), strict=True):
                    raw_scores_by_model[model_index][candidate.parent_id] = float(value)
        features = _gcd_patch_features(parent_patches)
        morphology = np.mean(
            np.stack(
                [row["model"].predict_proba(features)[:, 1] for row in _GCD_MORPH_PAYLOAD["models"]],
                axis=0,
            ),
            axis=0,
        )
        for candidate, value in zip(frame_candidates, morphology, strict=True):
            morphology_scores[candidate.parent_id] = float(value)
        del context_rows, context, context_mask, parent_patches, features, group_results
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint - 1:
                del frame_cache[cached_timepoint]

    deep_scores = _gcd_calibration_free_parent_scores(raw_scores_by_model, parent_ids)
    return deep_scores, morphology_scores, {
        "candidate_parents_scored": len(parent_ids),
        "candidate_frames_scored": len(by_time),
        "deep_member_count": len(_GCD_DEEP_MODELS),
        "gpu_groups_used": len(active_groups),
    }


def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    deep_scores, morphology_scores, inference_stats = _graph_context_scores_for_candidates(
        nodes_by_id, edges, dataset
    )
    recovered, recovery_stats = _apply_ranked_consensus(
        nodes_by_id,
        edges,
        deep_scores,
        morphology_scores,
        biological_geometry_minimum=_GCD_GEOMETRY_MINIMUM,
    )
    recovery_stats["candidate_parents_scored"] = inference_stats["candidate_parents_scored"]
    recovery_stats["gpu_groups_used"] = inference_stats["gpu_groups_used"]
    return recovered, {**inference_stats, **recovery_stats}

'''


CANDIDATE_EVIDENCE = (
    STRONG["CANDIDATE_EVIDENCE"]
    .replace("ema-strong-member-consensus-candidate-v2", RUN_ID)
    .replace("_RCD", "_GCD")
    .replace("_rcd", "_gcd")
    .replace('_GCD_POLICY["deep_policy"]', '_GCD_POLICY["graph_context_policy"]')
)


def build_notebook(consensus_root: Path) -> dict:
    return COMMON["transform_notebook"](
        consensus_root,
        dataset_builder_path=DATASET_BUILDER,
        watchdog_run_id=RUN_ID,
        keep_base_safe_divisions=True,
        attribution=ATTRIBUTION,
        model_setup_template=MODEL_SETUP_TEMPLATE,
        ranked_helpers=RANKED_HELPERS,
        candidate_evidence=CANDIDATE_EVIDENCE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source_metadata_path = SOURCE_DIR / "kernel-metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(source_metadata_path)
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
    notebook = build_notebook(args.consensus_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub EMA Graph Context Consensus v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "graph-context", "non-replica"],
        "dataset_sources": [*base_metadata["dataset_sources"], CONSENSUS_REF],
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
