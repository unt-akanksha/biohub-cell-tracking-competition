#!/usr/bin/env python
"""Build the attributed EMA plus learned-division submission candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle/biohub-clean-0927-repro-v1"
SOURCE_NOTEBOOK = SOURCE_DIR / "biohub-0-927-lb.ipynb"
SOURCE_NOTEBOOK_SHA256 = (
    "19d4c0f525be6325689af7bc37180098e5d19556d53636ed042e819b7fdeea24"
)
SOURCE_METADATA_SHA256 = (
    "37931e6f60a7987e0a0a48f2312f047ddf874a5e7ae2937801a1f3d0f7195266"
)
TARGET_ID = "biohub-ema-learned-division-candidate-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-learned-division-recovery-runtime-v1"
PRETRAIN_REF = "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"submission base drifted for {old!r}: expected {count}, saw {actual}"
        )
    return text.replace(old, new, count)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


ATTRIBUTION = """## Project candidate: EMA control plus learned division recovery

The detector, dual-seed ensemble, harmonic association, DeepCenter gap veto,
and constrained graph reconstruction are attributed to the public 0.927
lineage reproduced by `indarkarhana/biohub-clean-0-927-reproduction-v1`.
The running velocity EMA is independently reproduced from the documented idea
in `grafael/biohub-ct-0940-ema`.

The division stage is project-authored. Two 46.4M-parameter folds were trained
only on external ZebraHub data. A division-logit threshold was frozen on
ZSNS005 selection windows and opened once on disjoint ZSNS005 audit windows.
The public rule-based safe-division adder is disabled. The learned gate may add
only the nearest parent-free second daughter inside a bounded physical
neighborhood; it cannot change nodes, coordinates, or existing edges.
"""


MODEL_SETUP_TEMPLATE = r'''# Strictly load the external learned-division runtime and two v4 folds.
import hashlib as _ldr_hashlib
import json as _ldr_json
import math as _ldr_math
import sys as _ldr_sys

_LDR_RUNTIME_MANIFEST_SHA256 = "__RUNTIME_MANIFEST_SHA256__"

def _ldr_sha256(path):
    digest = _ldr_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_runtime_matches = []
for _path in INPUT_ROOT.rglob("RUNTIME_MANIFEST.json"):
    if _ldr_sha256(_path) != _LDR_RUNTIME_MANIFEST_SHA256:
        continue
    _payload = _ldr_json.loads(_path.read_text(encoding="utf-8"))
    if _payload.get("run_id") == "learned-division-recovery-runtime-v1":
        _runtime_matches.append(_path.parent)
if len(_runtime_matches) != 1:
    raise RuntimeError(f"Expected one learned-division runtime, saw {_runtime_matches}")
_ldr_runtime = _runtime_matches[0]
_ldr_manifest = _ldr_json.loads(
    (_ldr_runtime / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _ldr_manifest["files"].items():
    if _ldr_sha256(_ldr_runtime / _name) != _record["sha256"]:
        raise RuntimeError(f"Learned-division runtime source changed: {_name}")
_ldr_policy_path = _ldr_runtime / "division-recovery-policy.json"
if _ldr_sha256(_ldr_policy_path) != _ldr_manifest["policy_sha256"]:
    raise RuntimeError("Learned-division policy hash changed")
_ldr_policy_payload = _ldr_json.loads(_ldr_policy_path.read_text(encoding="utf-8"))
_ldr_threshold = float(_ldr_policy_payload["frozen_division_logit_threshold"])
if not (
    _ldr_policy_payload.get("status") == "accepted"
    and _ldr_policy_payload.get("run_id") == "external-division-recovery-policy-v1"
    and _ldr_policy_payload.get("audit_opened_after_threshold_freeze") is True
    and _ldr_policy_payload.get("competition_data_read") is False
    and _ldr_policy_payload.get("public_leaderboard_used_for_selection") is False
    and _ldr_policy_payload.get("submission_created") is False
    and _ldr_policy_payload.get("authorized_for_competition_graph_evaluation") is True
    and _ldr_policy_payload.get("authorized_for_submission") is False
    and _ldr_math.isfinite(_ldr_threshold)
):
    raise RuntimeError("Learned-division policy is ineligible")

_ldr_sys.path.insert(0, str(_ldr_runtime))
import torch as _ldr_torch
from learned_division_recovery import (
    DivisionRecoveryPolicy as _DivisionRecoveryPolicy,
    apply_learned_division_recovery as _apply_learned_division_recovery,
    discover_division_recovery_candidates as _discover_division_recovery_candidates,
)
from multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT as _LDR_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel as _LDRModel,
)
from patch_model import sample_physical_patches as _ldr_sample_physical_patches

_gpu_names = [
    _ldr_torch.cuda.get_device_name(index)
    for index in range(_ldr_torch.cuda.device_count())
]
if len(_gpu_names) != 2 or any("T4" not in name for name in _gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_gpu_names}")

_pretrain_matches = []
for _path in INPUT_ROOT.rglob("pretraining_terminal.json"):
    try:
        _payload = _ldr_json.loads(_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if (
        _payload.get("status") == "completed"
        and _payload.get("run_id") == "zebrahub-multiscale-contextual-pretrain-v1"
        and _payload.get("appearance_family")
        == "temporal_multiscale_contextual_pair_fusion_v4"
        and _payload.get("gpu_count") == 2
        and _payload.get("both_folds_improved") is True
        and _payload.get("competition_data_read") is False
        and _payload.get("public_leaderboard_used_for_selection") is False
        and _payload.get("submission_created") is False
    ):
        _pretrain_matches.append(_path.parent)
if len(_pretrain_matches) != 1:
    raise RuntimeError(f"Expected one verified v4 pretraining root, saw {_pretrain_matches}")
_ldr_pretrain = _pretrain_matches[0]
_LDR_MODELS = []
for _gpu_index, _fold in enumerate(("target_44b6", "target_6bba")):
    _checkpoint = _ldr_pretrain / _fold / "pretrained_model.pt"
    _expected_hash = _ldr_policy_payload["model_sha256"][_fold]
    if _ldr_sha256(_checkpoint) != _expected_hash:
        raise RuntimeError(f"V4 checkpoint changed for {_fold}")
    _device = _ldr_torch.device(f"cuda:{_gpu_index}")
    _model = _LDRModel().to(_device)
    _model.load_state_dict(
        _ldr_torch.load(_checkpoint, map_location=_device, weights_only=True),
        strict=True,
    )
    if sum(parameter.numel() for parameter in _model.parameters()) != _LDR_PARAMETER_COUNT:
        raise RuntimeError("Learned-division parameter inventory changed")
    _LDR_MODELS.append(_model.requires_grad_(False).eval())

_LDR_POLICY = _DivisionRecoveryPolicy(
    division_logit_threshold=_ldr_threshold,
    parent_distance_max_um=12.0,
    sister_distance_max_um=15.0,
    maximum_added_node_fraction=0.0025,
)
print({
    "learned_division_runtime": str(_ldr_runtime),
    "pretraining_root": str(_ldr_pretrain),
    "division_logit_threshold": _ldr_threshold,
    "gpu_names": _gpu_names,
})
'''


LEARNED_HELPERS = r'''
@_ldr_torch.inference_mode()
def _learned_division_logits_for_candidates(nodes_by_id, edges, dataset):
    candidates = _discover_division_recovery_candidates(
        nodes_by_id,
        edges,
        parent_distance_max_um=_LDR_POLICY.parent_distance_max_um,
        sister_distance_max_um=_LDR_POLICY.sister_distance_max_um,
    )
    parent_ids = sorted({candidate.parent_id for candidate in candidates})
    if not parent_ids:
        return {}, {"candidate_parents_scored": 0, "candidate_frames_scored": 0}
    by_time = {}
    for parent_id in parent_ids:
        by_time.setdefault(int(nodes_by_id[parent_id]["t"]), []).append(parent_id)
    maximum_time = max(int(node["t"]) for node in nodes_by_id.values())
    frame_cache = {}
    scores = {}
    for timepoint, frame_parent_ids in sorted(by_time.items()):
        frame_indices = [
            max(0, min(maximum_time, timepoint + offset))
            for offset in (-1, 0, 1)
        ]
        temporal = np.stack(
            [read_test_frame(dataset, index, frame_cache) for index in frame_indices],
            axis=0,
        )
        centers = np.asarray(
            [
                [
                    nodes_by_id[parent_id]["z"],
                    nodes_by_id[parent_id]["y"],
                    nodes_by_id[parent_id]["x"],
                ]
                for parent_id in frame_parent_ids
            ],
            dtype=np.float32,
        )
        fold_scores = []
        for gpu_index, model in enumerate(_LDR_MODELS):
            device = _ldr_torch.device(f"cuda:{gpu_index}")
            patches = _ldr_sample_physical_patches(
                _ldr_torch.as_tensor(temporal, device=device),
                centers,
                voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
                chunk_size=32,
            )
            divisions = []
            for start in range(0, len(patches), 32):
                with _ldr_torch.autocast(
                    device_type="cuda", dtype=_ldr_torch.float16
                ):
                    _embedding, logits = model(patches[start : start + 32])
                divisions.append(logits.float().cpu())
            fold_scores.append(_ldr_torch.cat(divisions).numpy())
            del patches
        ensemble = np.mean(np.stack(fold_scores, axis=0), axis=0)
        for parent_id, score in zip(frame_parent_ids, ensemble, strict=True):
            scores[parent_id] = float(score)
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint - 1:
                del frame_cache[cached_timepoint]
    return scores, {
        "candidate_parents_scored": len(parent_ids),
        "candidate_frames_scored": len(by_time),
    }


def _apply_external_learned_division_recovery(nodes_by_id, edges, dataset):
    logits, inference_stats = _learned_division_logits_for_candidates(
        nodes_by_id, edges, dataset
    )
    recovered, recovery_stats = _apply_learned_division_recovery(
        nodes_by_id, edges, logits, _LDR_POLICY
    )
    return recovered, {**inference_stats, **recovery_stats}

'''


LEARNED_APPLY = r'''
    if dataset is None:
        raise RuntimeError("Learned division recovery requires an explicit dataset")
    edges, learned_division_stats = _apply_external_learned_division_recovery(
        nodes_by_id, edges, dataset
    )
    for key, value in learned_division_stats.items():
        stats[f"learned_division_{key}"] = value
    print(
        f"  [{dataset}] learned division recovery: "
        f"candidates={learned_division_stats['geometric_candidates']} "
        f"scored={learned_division_stats['candidate_parents_scored']} "
        f"added={learned_division_stats['added_edges']}"
    )

'''


CANDIDATE_EVIDENCE = r'''# Emit hash-bound evidence for the external promotion gate.
_evidence_submission = Path("/kaggle/working/submission.csv")
_evidence_stats = pd.read_csv(RUN_STATS_PATH)
_evidence_validator = pd.read_csv(VALIDATOR_STATS_PATH)
_evidence_weight = float(_evidence_validator["weight"].sum())
if _evidence_weight <= 0:
    raise RuntimeError("Candidate validator emitted no weighted rows")
_evidence_adjusted_edge = float(
    (
        _evidence_validator["adjusted_edge_jaccard"]
        * _evidence_validator["weight"]
    ).sum()
    / _evidence_weight
)
_evidence_div_tp = int(_evidence_validator["div_tp"].sum())
_evidence_div_fp = int(_evidence_validator["div_fp"].sum())
_evidence_div_fn = int(_evidence_validator["div_fn"].sum())
_evidence_div_denominator = (
    _evidence_div_tp + _evidence_div_fp + _evidence_div_fn
)
_evidence_division = (
    _evidence_div_tp / _evidence_div_denominator
    if _evidence_div_denominator
    else 0.0
)
_evidence = {
    "schema_version": 1,
    "status": "completed_pending_external_promotion_gate",
    "run_id": "ema-learned-division-candidate-v1",
    "target_public_score": 0.945,
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "external_training_only": True,
    "runtime_manifest_sha256": _LDR_RUNTIME_MANIFEST_SHA256,
    "policy_sha256": _ldr_manifest["policy_sha256"],
    "pretraining_model_sha256": _ldr_policy_payload["model_sha256"],
    "safe_division_heuristic_disabled": (
        os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS") == "0"
    ),
    "learned_geometric_candidates": int(
        _evidence_stats["learned_division_geometric_candidates"].sum()
    ),
    "learned_candidate_parents_scored": int(
        _evidence_stats["learned_division_candidate_parents_scored"].sum()
    ),
    "learned_edges_added": int(
        _evidence_stats["learned_division_added_edges"].sum()
    ),
    "learned_reassignments": int(
        _evidence_stats["learned_division_reassignment_performed"].sum()
    ),
    "learned_node_or_coordinate_changes": int(
        _evidence_stats["learned_division_node_or_coordinate_changes"].sum()
    ),
    "validator_adjusted_edge_jaccard": _evidence_adjusted_edge,
    "validator_division_tp": _evidence_div_tp,
    "validator_division_fp": _evidence_div_fp,
    "validator_division_fn": _evidence_div_fn,
    "validator_division_jaccard": _evidence_division,
    "validator_proxy_score": _evidence_adjusted_edge + 0.10 * _evidence_division,
    "submission_sha256": _ldr_sha256(_evidence_submission),
    "competition_submission_performed": False,
    "authorized_for_submission": False,
}
_evidence_path = Path("/kaggle/working/candidate_evidence.json")
_evidence_path.write_text(
    _ldr_json.dumps(_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_ldr_json.dumps(_evidence, indent=2, sort_keys=True))
'''


def transform_notebook(runtime_root: Path) -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Attributed public-control notebook changed")
    runtime_builder = runpy.run_path(
        str(ROOT / "scripts/build-learned-division-recovery-runtime.py")
    )
    runtime = runtime_builder["verify_runtime"](runtime_root)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="ascii"))
    watchdog = "".join(notebook["cells"][0]["source"])
    watchdog = replace_exact(
        watchdog,
        "_BIOHUB_RUN_ID = 'public-0927-clean-repro-v2'",
        "_BIOHUB_RUN_ID = 'ema-learned-division-candidate-v1'",
    )
    notebook["cells"][0]["source"] = watchdog.splitlines(keepends=True)

    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if 'os.environ["BIOHUB_DET_THRESHOLD"]' in "".join(cell.get("source", []))
    )
    config = "".join(notebook["cells"][config_index]["source"])
    config = replace_exact(
        config,
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"',
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"\n'
        'os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "1.0"\n'
        'os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"',
    )
    notebook["cells"][config_index]["source"] = config.splitlines(keepends=True)

    guard_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Configuration guard: assert" in "".join(cell.get("source", []))
    )
    guard = "".join(notebook["cells"][guard_index]["source"])
    guard = replace_exact(
        guard,
        '    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.30,',
        '    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.30,\n'
        '    "BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT": 1.0,',
    )
    guard = replace_exact(
        guard,
        '    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",',
        '    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",\n'
        '    "BIOHUB_OUTPUT_SAFE_DIVISIONS": "0",',
    )
    guard = guard.replace(
        "Single model-level change: harmonic mutual-support association fusion",
        "Candidate changes: EMA motion plus externally learned division recovery",
    )
    notebook["cells"][guard_index]["source"] = guard.splitlines(keepends=True)

    post_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "def motion_relink_edges(" in "".join(cell.get("source", []))
        and "def filter_output_graph(" in "".join(cell.get("source", []))
    )
    post = "".join(notebook["cells"][post_index]["source"])
    post = replace_exact(
        post,
        "    predecessor_position_um: dict[int, np.ndarray] = {}\n"
        "    selected_edges: list[dict[str, object]] = []",
        "    predecessor_position_um: dict[int, np.ndarray] = {}\n"
        "    velocity_um: dict[int, np.ndarray] = {}\n"
        "    velocity_ema_alpha = 0.4\n"
        "    selected_edges: list[dict[str, object]] = []",
    )
    post = replace_exact(
        post,
        "            prev_pos = predecessor_position_um.get(source_id)\n"
        "            if prev_pos is None:\n"
        "                predicted = source_pos\n"
        "            else:\n"
        "                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)",
        "            prev_pos = predecessor_position_um.get(source_id)\n"
        "            velocity = velocity_um.get(source_id)\n"
        "            if velocity is not None:\n"
        "                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * velocity\n"
        "            elif prev_pos is None:\n"
        "                predicted = source_pos\n"
        "            else:\n"
        "                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)",
    )
    post = replace_exact(
        post,
        "            predecessor_position_um[target_id] = position_um[source_id]",
        "            predecessor_position_um[target_id] = position_um[source_id]\n"
        "            step_velocity = position_um[target_id] - position_um[source_id]\n"
        "            previous_velocity = velocity_um.get(source_id)\n"
        "            velocity_um[target_id] = (\n"
        "                step_velocity if previous_velocity is None\n"
        "                else velocity_ema_alpha * step_velocity\n"
        "                + (1.0 - velocity_ema_alpha) * previous_velocity\n"
        "            )",
    )
    post = replace_exact(
        post,
        "def filter_output_graph(\n",
        LEARNED_HELPERS + "def filter_output_graph(\n",
    )
    post = replace_exact(
        post,
        "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
        LEARNED_APPLY
        + "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
    )
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)

    inference_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Fail fast instead of silently running volumetric inference on CPU"
        in "".join(cell.get("source", []))
    )
    setup = MODEL_SETUP_TEMPLATE.replace(
        "__RUNTIME_MANIFEST_SHA256__", str(runtime["manifest_sha256"])
    )
    notebook["cells"][inference_index + 1 : inference_index + 1] = [
        markdown_cell(ATTRIBUTION),
        code_cell(setup),
    ]
    terminal_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if '_biohub_write_terminal("completed")'
        in "".join(cell.get("source", []))
    )
    notebook["cells"][terminal_index:terminal_index] = [
        code_cell(CANDIDATE_EVIDENCE)
    ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source_metadata_path = SOURCE_DIR / "kernel-metadata.json"
    if sha256_file(source_metadata_path) != SOURCE_METADATA_SHA256:
        raise RuntimeError("Attributed public-control metadata changed")
    if TARGET_DIR.exists():
        if not any(TARGET_DIR.iterdir()):
            TARGET_DIR.rmdir()
        elif not args.replace:
            raise FileExistsError(TARGET_DIR)
        else:
            shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = transform_notebook(args.runtime_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub EMA Learned Division Candidate v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "learned-division", "non-replica"],
        "dataset_sources": [*base_metadata["dataset_sources"], RUNTIME_REF],
        "kernel_sources": [PRETRAIN_REF],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
