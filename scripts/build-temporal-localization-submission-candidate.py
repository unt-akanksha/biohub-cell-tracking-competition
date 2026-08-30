#!/usr/bin/env python
"""Build the attributed EMA control plus temporal-localization candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py"))
DATASET_BUILDER = ROOT / "scripts/build-temporal-localization-candidate-dataset.py"
SOURCE_DIR = ROOT / "kaggle/biohub-clean-0927-repro-v1"
SOURCE_METADATA_SHA256 = COMMON["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-ema-temporal-localization-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-temporal-localization-consensus-v1"
RUN_ID = "ema-temporal-localization-candidate-v1"


ATTRIBUTION = """## Project candidate: attributed EMA control plus temporal localization

The detector, harmonic association, gap handling, safe-division rule, and graph
reconstruction retain their original public attribution. No public prediction
is copied. The additional coordinate stage is project-authored and cannot add
or remove nodes or alter lineage edges.

Every attached 71.25M-parameter temporal ConvNeXt3D/axial member trained on
Synthetic256 plus a fixed 25% train-only real-image replay stream. Each member
passed independent global and division-critical gates on both domains, followed
by sealed audits on both domains. Ensemble membership contains every dual-domain
eligible member and was frozen before the already-opened real development probe.
The real probe supplied one transfer gate only; it did not search thresholds,
weights, or member subsets. At
inference, at least three members must agree in direction and within 1.5 um,
predict low uncertainty, and classify the current coordinate as outside the
5 um safe radius. A global 10% move-fraction gate fails closed. The notebook
requires exactly two T4 GPUs and partitions accepted members across both.
"""


MODEL_SETUP_TEMPLATE = r'''# Strictly load the synthetic-gated localization runtime.
import concurrent.futures as _tlc_futures
import hashlib as _tlc_hashlib
import json as _tlc_json
import sys as _tlc_sys

_TLC_MANIFEST_SHA256 = "__MANIFEST_SHA256__"
_TLC_INPUT_ROOT = Path("/kaggle/input")

def _tlc_sha256(path):
    digest = _tlc_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_tlc_matches = []
for _path in _TLC_INPUT_ROOT.rglob("TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json"):
    if _tlc_sha256(_path) != _TLC_MANIFEST_SHA256:
        continue
    _payload = _tlc_json.loads(_path.read_text(encoding="utf-8"))
    if _payload.get("run_id") == "competition-temporal-localization-consensus-dataset-v1":
        _tlc_matches.append(_path.parent)
if len(_tlc_matches) != 1:
    raise RuntimeError(f"Expected one temporal-localization runtime, saw {_tlc_matches}")
_TLC_ROOT = _tlc_matches[0]
_TLC_MANIFEST = _tlc_json.loads(
    (_TLC_ROOT / "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _TLC_MANIFEST["files"].items():
    if _record.get("path") != _name or _tlc_sha256(_TLC_ROOT / _name) != _record.get("sha256"):
        raise RuntimeError(f"Temporal-localization runtime changed: {_name}")

_TLC_POLICY = _tlc_json.loads(
    (_TLC_ROOT / "temporal-localization-consensus-policy.json").read_text(encoding="utf-8")
)
_TLC_DEVELOPMENT = _tlc_json.loads(
    (_TLC_ROOT / "evidence/real-development-probe.json").read_text(encoding="utf-8")
)
_TLC_MEMBERS = _TLC_POLICY.get("localization_members", [])
if not (
    _TLC_POLICY.get("schema_version") == 1
    and _TLC_POLICY.get("status") == "development_accepted"
    and _TLC_POLICY.get("run_id") == "competition-temporal-localization-consensus-policy-v1"
    and 3 <= len(_TLC_MEMBERS) <= 4
    and _TLC_POLICY.get("localization_member_count") == len(_TLC_MEMBERS)
    and len({row["model_sha256"] for row in _TLC_MEMBERS}) == len(_TLC_MEMBERS)
    and all(
        row.get("model_sha256") == _tlc_sha256(_TLC_ROOT / row["path"])
        and row.get("parameter_count") == 71_249_805
        and row.get("selection_gate_passed") is True
        and row.get("audit_gate_passed") is True
        and row.get("division_critical_selection_gate_passed") is True
        and row.get("division_critical_audit_gate_passed") is True
        and row.get("real_selection_gate_passed") is True
        and row.get("real_audit_gate_passed") is True
        and row.get("real_division_critical_selection_gate_passed") is True
        and row.get("real_division_critical_audit_gate_passed") is True
        and row.get("serialized_checkpoint_selection_gate_passed") is True
        for row in _TLC_MEMBERS
    )
    and _TLC_POLICY.get("ensemble_policy") == "equal_mean_all_dual_domain_eligible_members"
    and float(_TLC_POLICY.get("real_replay_probability")) == 0.25
    and _TLC_POLICY.get("minimum_members") == 3
    and float(_TLC_POLICY.get("blend")) == 0.75
    and float(_TLC_POLICY.get("minimum_correction_um")) == 2.0
    and float(_TLC_POLICY.get("maximum_correction_um")) == 9.5
    and float(_TLC_POLICY.get("maximum_member_disagreement_um")) == 1.5
    and float(_TLC_POLICY.get("maximum_predicted_sigma_um")) == 2.5
    and float(_TLC_POLICY.get("maximum_safe_probability")) == 0.35
    and float(_TLC_POLICY.get("minimum_direction_cosine")) == 0.8
    and float(_TLC_POLICY.get("maximum_move_fraction")) == 0.1
    and float(_TLC_POLICY.get("minimum_forced_division_critical_fraction")) == 0.25
    and _TLC_POLICY.get("division_critical_selection_gate_required") is True
    and _TLC_POLICY.get("division_critical_audit_gate_required") is True
    and _TLC_POLICY.get("real_selection_gate_required") is True
    and _TLC_POLICY.get("real_audit_gate_required") is True
    and _TLC_POLICY.get("real_division_critical_selection_gate_required") is True
    and _TLC_POLICY.get("real_division_critical_audit_gate_required") is True
    and _TLC_POLICY.get("node_count_preserving") is True
    and _TLC_POLICY.get("topology_preserving") is True
    and _TLC_POLICY.get("exact_two_t4_required") is True
    and _TLC_POLICY.get("model_subset_searched_on_audit") is False
    and _TLC_POLICY.get("weights_searched_on_development") is False
    and _TLC_POLICY.get("threshold_searched_on_development") is False
    and _TLC_POLICY.get("public_leaderboard_used_for_selection") is False
    and _TLC_POLICY.get("authorized_for_full_candidate_evaluation") is True
    and _TLC_POLICY.get("authorized_for_submission") is False
    and _TLC_DEVELOPMENT.get("status") == "development_passed"
    and _TLC_DEVELOPMENT.get("run_id") == "temporal-node-localizer-real-development-v1"
    and _TLC_DEVELOPMENT.get("gate", {}).get("passed") is True
    and _TLC_DEVELOPMENT.get("policy_or_member_selection_performed") is False
    and _TLC_DEVELOPMENT.get("competition_test_data_read") is False
    and _TLC_DEVELOPMENT.get("public_leaderboard_used_for_selection") is False
    and _TLC_DEVELOPMENT.get("authorized_for_submission") is False
):
    raise RuntimeError("Temporal-localization source evidence is ineligible")
for _member in _TLC_MEMBERS:
    _terminal = _tlc_json.loads((_TLC_ROOT / _member["terminal_path"]).read_text(encoding="utf-8"))
    if not (
        _terminal.get("status") == "completed"
        and _terminal.get("run_id") == "synthetic256-real-replay-temporal-node-localizer-v2"
        and _terminal.get("model_sha256") == _member["model_sha256"]
        and _terminal.get("checkpoint_frozen_before_audit") is True
        and _terminal.get("selection_gate_passed") is True
        and _terminal.get("audit_gate_passed") is True
        and _terminal.get("division_critical_selection_gate_passed") is True
        and _terminal.get("division_critical_audit_gate_passed") is True
        and _terminal.get("real_selection_gate_passed") is True
        and _terminal.get("real_audit_gate_passed") is True
        and _terminal.get("real_division_critical_selection_gate_passed") is True
        and _terminal.get("real_division_critical_audit_gate_passed") is True
        and _terminal.get("serialized_checkpoint_selection_gate_passed") is True
        and float(_terminal.get("real_replay_probability")) == 0.25
        and _terminal.get("competition_train_data_read") is True
        and _terminal.get("competition_test_data_read") is False
        and _terminal.get("public_leaderboard_used_for_selection") is False
    ):
        raise RuntimeError("Temporal-localization member terminal changed")

_tlc_sys.path.insert(0, str(_TLC_ROOT))
import torch as _tlc_torch
from research.temporal_contrastive.patch_model import sample_physical_patches as _tlc_sample_physical_patches
from research.temporal_localization.consensus import select_consensus_offsets as _tlc_select_consensus_offsets
from research.temporal_localization.inference import MovieGraphArrays as _TLCMovieGraphArrays
from research.temporal_localization.inference import graph_motion_features as _tlc_graph_motion_features
from research.temporal_localization.model import TemporalNodeLocalizationModel as _TLCModel

_tlc_gpu_names = [_tlc_torch.cuda.get_device_name(index) for index in range(_tlc_torch.cuda.device_count())]
if len(_tlc_gpu_names) != 2 or any("T4" not in name for name in _tlc_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_tlc_gpu_names}")
_TLC_DEVICES = [_tlc_torch.device("cuda:0"), _tlc_torch.device("cuda:1")]
_TLC_MODELS_BY_DEVICE = [[], []]
_TLC_MODELS = []
for _index, _member in enumerate(_TLC_MEMBERS):
    _device_index = _index % 2
    _model = _TLCModel().to(_TLC_DEVICES[_device_index])
    _model.load_state_dict(
        _tlc_torch.load(_TLC_ROOT / _member["path"], map_location=_TLC_DEVICES[_device_index], weights_only=True),
        strict=True,
    )
    if sum(parameter.numel() for parameter in _model.parameters()) != 71_249_805:
        raise RuntimeError("Temporal-localization parameter inventory changed")
    _model.requires_grad_(False).eval()
    _TLC_MODELS_BY_DEVICE[_device_index].append((_index, _model))
    _TLC_MODELS.append(_model)
print({
    "localization_root": str(_TLC_ROOT),
    "localization_members": len(_TLC_MODELS),
    "members_per_gpu": [len(row) for row in _TLC_MODELS_BY_DEVICE],
    "parameters_per_member": 71_249_805,
    "gpu_names": _tlc_gpu_names,
})
'''


LOCALIZATION_HELPERS = r'''
def _tlc_graph_arrays(nodes_by_id, edges):
    node_ids = np.asarray(sorted(nodes_by_id), dtype=np.int64)
    rows = [nodes_by_id[int(node_id)] for node_id in node_ids]
    edge_rows = np.asarray(
        [[int(row["source_id"]), int(row["target_id"])] for row in edges],
        dtype=np.int64,
    ).reshape(-1, 2)
    return _TLCMovieGraphArrays(
        node_ids=node_ids,
        times=np.asarray([int(row["t"]) for row in rows], dtype=np.int64),
        coordinates_zyx_voxel=np.asarray(
            [[float(row["z"]), float(row["y"]), float(row["x"])] for row in rows],
            dtype=np.float32,
        ),
        edges_by_node_id=edge_rows,
    )


@_tlc_torch.inference_mode()
def _tlc_score_device_group(device_index, graph, dataset, frames, spatial_shape, timepoints):
    indexed_models = _TLC_MODELS_BY_DEVICE[device_index]
    device = _TLC_DEVICES[device_index]
    selected_rows = np.concatenate(
        [np.flatnonzero(graph.times == frame).astype(np.int64) for frame in frames]
    ) if frames else np.empty((0,), dtype=np.int64)
    model_offsets = [np.empty((len(selected_rows), 3), dtype=np.float32) for _ in indexed_models]
    model_sigma = [np.empty((len(selected_rows), 3), dtype=np.float32) for _ in indexed_models]
    model_safe = [np.empty((len(selected_rows),), dtype=np.float32) for _ in indexed_models]
    cursor = 0
    frame_cache = {}
    for frame in frames:
        frame_rows = np.flatnonzero(graph.times == frame).astype(np.int64)
        if not len(frame_rows):
            continue
        context = _tlc_torch.as_tensor(
            np.stack([read_test_frame(dataset, index, frame_cache) for index in (frame - 1, frame, frame + 1)]),
            dtype=_tlc_torch.float32,
            device=device,
        )
        features = _tlc_graph_motion_features(
            graph,
            frame_rows,
            voxel_size_zyx_um=np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32),
            spatial_shape_zyx=spatial_shape,
            timepoints=timepoints,
        )
        for start in range(0, len(frame_rows), 16):
            batch_rows = frame_rows[start : start + 16]
            patches = _tlc_sample_physical_patches(
                context,
                _tlc_torch.as_tensor(graph.coordinates_zyx_voxel[batch_rows], dtype=_tlc_torch.float32, device=device),
                voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
                output_shape_zyx=(17, 17, 17),
                half_extent_zyx_um=(12.0, 12.0, 12.0),
                chunk_size=len(batch_rows),
            )
            graph_features = _tlc_torch.as_tensor(
                features[start : start + 16], dtype=_tlc_torch.float32, device=device
            )
            stop = cursor + len(batch_rows)
            for local_index, (_model_index, model) in enumerate(indexed_models):
                with _tlc_torch.autocast(device_type="cuda", dtype=_tlc_torch.float16):
                    offsets, log_variance, safe = model(patches, graph_features)
                model_offsets[local_index][cursor:stop] = offsets.float().cpu().numpy()
                model_sigma[local_index][cursor:stop] = _tlc_torch.exp(0.5 * log_variance.float()).cpu().numpy()
                model_safe[local_index][cursor:stop] = safe.float().cpu().numpy()
            cursor = stop
            del patches, graph_features
        for cached_frame in list(frame_cache):
            if cached_frame < frame - 1:
                del frame_cache[cached_frame]
    if cursor != len(selected_rows):
        raise RuntimeError("Temporal-localization device group lost graph rows")
    return (
        [row[0] for row in indexed_models],
        selected_rows,
        model_offsets,
        model_sigma,
        model_safe,
    )


def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    if dataset is None:
        raise RuntimeError("Temporal localization requires an explicit dataset")
    graph = _tlc_graph_arrays(nodes_by_id, edges)
    meta = _tlc_json.loads((TEST_DIR / f"{dataset}.zarr" / "0" / "zarr.json").read_text())
    shape = tuple(int(value) for value in meta["shape"])
    timepoints = shape[0]
    spatial_shape = shape[1:]
    frames = tuple(
        sorted(
            int(value)
            for value in np.unique(graph.times)
            if 0 < int(value) < timepoints - 1
        )
    )
    selected_rows = np.concatenate(
        [np.flatnonzero(graph.times == frame).astype(np.int64) for frame in frames]
    ) if frames else np.empty((0,), dtype=np.int64)
    if not len(selected_rows):
        return edges, {
            "geometric_candidates": 0,
            "geometry_eligible_candidates": 0,
            "ranking_agreed": 0,
            "added_edges": 0,
            "localization_candidate_nodes": 0,
            "localization_nodes_moved": 0,
            "localization_boundary_rejected": 0,
            "localization_rounded_coordinate_changes": 0,
            "localization_global_gate_failures": 0,
            "localization_gpu_groups_used": 0,
            "localization_node_count_changes": 0,
            "localization_edge_changes": 0,
        }
    active_groups = [index for index, rows in enumerate(_TLC_MODELS_BY_DEVICE) if rows]
    with _tlc_futures.ThreadPoolExecutor(max_workers=len(active_groups)) as executor:
        group_results = list(
            executor.map(
                lambda device_index: _tlc_score_device_group(
                    device_index, graph, dataset, frames, spatial_shape, timepoints
                ),
                active_groups,
            )
        )
    member_offsets = [None] * len(_TLC_MODELS)
    member_sigma = [None] * len(_TLC_MODELS)
    member_safe = [None] * len(_TLC_MODELS)
    for model_indices, rows, offsets, sigma, safe in group_results:
        if not np.array_equal(rows, selected_rows):
            raise RuntimeError("Temporal-localization GPU groups returned different node rows")
        for local_index, model_index in enumerate(model_indices):
            member_offsets[model_index] = offsets[local_index]
            member_sigma[model_index] = sigma[local_index]
            member_safe[model_index] = safe[local_index]
    if any(value is None for value in (*member_offsets, *member_sigma, *member_safe)):
        raise RuntimeError("Temporal-localization ensemble member output is missing")
    consensus = _tlc_select_consensus_offsets(
        np.stack(member_offsets), np.stack(member_sigma), np.stack(member_safe)
    )
    scale = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
    moved = 0
    boundary_rejected = 0
    rounded_changes = 0
    for inventory_row, graph_row in enumerate(selected_rows.tolist()):
        if not consensus.selected[inventory_row]:
            continue
        coordinate = graph.coordinates_zyx_voxel[graph_row] + consensus.offsets_um[inventory_row] / scale
        if np.any(coordinate < 0.0) or np.any(coordinate >= np.asarray(spatial_shape, dtype=np.float32)):
            boundary_rejected += 1
            continue
        node_id = int(graph.node_ids[graph_row])
        original = graph.coordinates_zyx_voxel[graph_row]
        if tuple(np.rint(coordinate).astype(np.int64)) != tuple(np.rint(original).astype(np.int64)):
            rounded_changes += 1
        nodes_by_id[node_id]["z"] = float(coordinate[0])
        nodes_by_id[node_id]["y"] = float(coordinate[1])
        nodes_by_id[node_id]["x"] = float(coordinate[2])
        moved += 1
    del group_results, member_offsets, member_sigma, member_safe
    return edges, {
        "geometric_candidates": len(selected_rows),
        "geometry_eligible_candidates": int(consensus.selected.sum()),
        "ranking_agreed": int(consensus.selected.sum()),
        "added_edges": 0,
        "localization_candidate_nodes": len(selected_rows),
        "localization_nodes_moved": moved,
        "localization_boundary_rejected": boundary_rejected,
        "localization_rounded_coordinate_changes": rounded_changes,
        "localization_global_gate_failures": int(not consensus.global_gate_passed),
        "localization_gpu_groups_used": len(active_groups),
        "localization_node_count_changes": 0,
        "localization_edge_changes": 0,
    }

'''


CANDIDATE_EVIDENCE = r'''# Emit hash-bound evidence for the external localization promotion gate.
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
_evidence_division = _evidence_div_tp / _evidence_div_denominator if _evidence_div_denominator else 0.0
_evidence = {
    "schema_version": 1,
    "status": "completed_pending_external_promotion_gate",
    "run_id": "ema-temporal-localization-candidate-v1",
    "target_public_score": 0.945,
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "localization_family": "temporal_convnext_axial_node_localizer_v1",
    "localization_member_count": len(_TLC_MODELS),
    "parameters_per_member": 71_249_805,
    "localization_model_sha256": [row["model_sha256"] for row in _TLC_MEMBERS],
    "runtime_manifest_sha256": _TLC_MANIFEST_SHA256,
    "node_count_preserving": True,
    "topology_preserving": True,
    "model_subset_searched_on_audit": False,
    "weights_searched_on_development": False,
    "threshold_searched_on_development": False,
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
_evidence_path.write_text(_tlc_json.dumps(_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(_tlc_json.dumps(_evidence, indent=2, sort_keys=True))
'''


def build_notebook(runtime_root: Path) -> dict:
    notebook = COMMON["transform_notebook"](
        runtime_root,
        dataset_builder_path=DATASET_BUILDER,
        watchdog_run_id=RUN_ID,
        keep_base_safe_divisions=True,
        attribution=ATTRIBUTION,
        model_setup_template=MODEL_SETUP_TEMPLATE,
        ranked_helpers=LOCALIZATION_HELPERS,
        candidate_evidence=CANDIDATE_EVIDENCE,
    )
    watchdog = "".join(notebook["cells"][0]["source"])
    replacements = {
        "# Biohub quota watchdog: 3600 s declared budget, 600 s safety margin.":
            "# Biohub quota watchdog: 39600 s declared budget, 1200 s safety margin.",
        '"declared_budget_seconds": 3600,': '"declared_budget_seconds": 39600,',
        '"safety_margin_seconds": 600,': '"safety_margin_seconds": 1200,',
        "_BIOHUB_TIMER = _biohub_threading.Timer(3000, _biohub_budget_expired)":
            "_BIOHUB_TIMER = _biohub_threading.Timer(38400, _biohub_budget_expired)",
        'print("Biohub watchdog armed: hard stop after 3000 seconds.")':
            'print("Biohub watchdog armed: hard stop after 38400 seconds.")',
    }
    for old, new in replacements.items():
        if watchdog.count(old) != 1:
            raise RuntimeError(f"Attributed watchdog contract changed: {old}")
        watchdog = watchdog.replace(old, new)
    notebook["cells"][0]["source"] = watchdog.splitlines(keepends=True)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
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
    notebook = build_notebook(args.runtime_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub EMA Temporal Localization v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "localization", "non-replica"],
        "dataset_sources": [*base_metadata["dataset_sources"], RUNTIME_REF],
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
