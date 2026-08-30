#!/usr/bin/env python
"""Build the additive EMA plus admitted strong-member consensus candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(
    str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py")
)
DATASET_BUILDER = ROOT / "scripts/build-strong-member-consensus-division-dataset.py"
SOURCE_DIR = ROOT / "kaggle/biohub-clean-0927-repro-v1"
SOURCE_METADATA_SHA256 = COMMON["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-ema-strong-member-candidate-v2"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
CONSENSUS_REF = "indarkarhana/biohub-strong-member-consensus-division-v2"
RUN_ID = "ema-strong-member-consensus-candidate-v2"


ATTRIBUTION = """## Project candidate: EMA control plus admitted strong-member consensus

The detector, dual-seed ensemble, harmonic association, DeepCenter gap veto,
safe-division rule, and constrained graph reconstruction are attributed to the
public 0.927 lineage reproduced by
`indarkarhana/biohub-clean-0-927-reproduction-v1`. The running velocity EMA is
independently reproduced from the documented idea in
`grafael/biohub-ct-0940-ema`.

The additional division stage is project-authored and selected without
leaderboard feedback. Every deep voter is a separately optimized 46.4M-
parameter temporal multiscale network that passed pooled, per-embryo, and
zero-false-positive selection gates. Byte-identical checkpoints cannot vote.
The frozen deep policy is either the equal average of within-movie percentile
ranks from all admitted networks or the single strongest predeclared network.
It must agree with an independently trained 132-feature temporal morphology
ensemble before adding at most one parent-free edge per movie. No absolute
deep threshold, ensemble-weight search, reassignment, node edit, or coordinate
edit is used. The clean base safe-division rule remains enabled.
"""


MODEL_SETUP_TEMPLATE = r'''# Strictly load the additive strong-member runtime.
import hashlib as _rcd_hashlib
import json as _rcd_json
import subprocess as _rcd_subprocess
import sys as _rcd_sys

_RCD_MANIFEST_SHA256 = "__MANIFEST_SHA256__"
_RCD_INPUT_ROOT = Path("/kaggle/input")

def _rcd_sha256(path):
    digest = _rcd_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_rcd_matches = []
for _path in _RCD_INPUT_ROOT.rglob("STRONG_MEMBER_CONSENSUS_MANIFEST.json"):
    if _rcd_sha256(_path) != _RCD_MANIFEST_SHA256:
        continue
    _payload = _rcd_json.loads(_path.read_text(encoding="utf-8"))
    if _payload.get("run_id") == "competition-strong-member-consensus-division-dataset-v2":
        _rcd_matches.append(_path.parent)
if len(_rcd_matches) != 1:
    raise RuntimeError(f"Expected one strong-member runtime, saw {_rcd_matches}")
_RCD_ROOT = _rcd_matches[0]
_RCD_MANIFEST = _rcd_json.loads(
    (_RCD_ROOT / "STRONG_MEMBER_CONSENSUS_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _RCD_MANIFEST["files"].items():
    if _rcd_sha256(_RCD_ROOT / _name) != _record["sha256"]:
        raise RuntimeError(f"Strong-member runtime changed: {_name}")

_RCD_POLICY = _rcd_json.loads(
    (_RCD_ROOT / "strong-member-consensus-policy.json").read_text(encoding="utf-8")
)
_RCD_DEVELOPMENT = _rcd_json.loads(
    (_RCD_ROOT / "development_evidence.json").read_text(encoding="utf-8")
)
_RCD_DEEP_SELECTION = _rcd_json.loads(
    (_RCD_ROOT / "deep_selection_terminal.json").read_text(encoding="utf-8")
)
_RCD_DEEP_PROBE = _rcd_json.loads(
    (_RCD_ROOT / "deep_development_probe.json").read_text(encoding="utf-8")
)
_RCD_MORPH_TERMINAL = _rcd_json.loads(
    (_RCD_ROOT / "morphology_training_terminal.json").read_text(encoding="utf-8")
)
_RCD_DEEP_MEMBERS = _RCD_POLICY.get("deep_members", [])
if not (
    _RCD_POLICY.get("schema_version") == 1
    and _RCD_POLICY.get("status") == "development_accepted"
    and _RCD_POLICY.get("run_id")
    == "competition-strong-member-consensus-division-policy-v2"
    and _RCD_POLICY.get("deep_policy")
    in {"equal_rank_admitted_ensemble", "strongest_individual_rank"}
    and 1 <= len(_RCD_DEEP_MEMBERS) <= 16
    and _RCD_POLICY.get("deep_member_count") == len(_RCD_DEEP_MEMBERS)
    and len({row["model_sha256"] for row in _RCD_DEEP_MEMBERS})
    == len(_RCD_DEEP_MEMBERS)
    and all(
        row.get("model_sha256") == _rcd_sha256(_RCD_ROOT / row["path"])
        and row.get("parameter_count") == 46_386_607
        and row.get("trainable_parameters") == 25_178_047
        for row in _RCD_DEEP_MEMBERS
    )
    and _RCD_POLICY.get("base_safe_division_heuristic_enabled") is True
    and _RCD_POLICY.get("external_policy_additive_only") is True
    and _RCD_POLICY.get("absolute_threshold_used") is False
    and _RCD_POLICY.get("maximum_added_edges_per_movie") == 1
    and float(_RCD_POLICY.get("biological_geometry_minimum")) == 3.0
    and _RCD_POLICY.get("morphology_model_sha256")
    == _rcd_sha256(_RCD_ROOT / "morphology_division_model.joblib")
    and _RCD_POLICY.get("sklearn_version") == "1.9.0"
    and _RCD_POLICY.get("sklearn_wheel_sha256")
    == _rcd_sha256(
        _RCD_ROOT
        / "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    and _RCD_DEVELOPMENT.get("status") == "development_positive"
    and _RCD_DEVELOPMENT.get("selected", 0) >= 1
    and _RCD_DEVELOPMENT.get("tp") == _RCD_DEVELOPMENT.get("selected")
    and _RCD_DEVELOPMENT.get("fp") == 0
    and _RCD_DEVELOPMENT.get("deep_probe_sha256")
    == _rcd_sha256(_RCD_ROOT / "deep_development_probe.json")
    and _RCD_DEVELOPMENT.get("absolute_threshold_used") is False
    and _RCD_DEVELOPMENT.get("authorized_for_full_candidate_evaluation") is True
    and _RCD_DEVELOPMENT.get("authorized_for_submission") is False
    and _RCD_DEEP_SELECTION.get("authorized_for_development_probe") is True
    and _RCD_DEEP_SELECTION.get("competition_test_data_read") is False
    and _RCD_DEEP_SELECTION.get("public_leaderboard_used_for_selection") is False
    and _RCD_DEEP_SELECTION.get("submission_created") is False
    and _RCD_DEEP_PROBE.get("status") == "development_probe_complete"
    and _RCD_DEEP_PROBE.get("member_count") == len(_RCD_DEEP_MEMBERS)
    and _RCD_DEEP_PROBE.get("absolute_threshold_used") is False
    and _RCD_DEEP_PROBE.get("weights_searched_on_probe") is False
    and _RCD_DEEP_PROBE.get("model_subset_searched_on_probe") is False
    and _RCD_DEEP_PROBE.get("competition_test_data_read") is False
    and _RCD_DEEP_PROBE.get("public_leaderboard_used_for_selection") is False
    and _RCD_MORPH_TERMINAL.get("status") == "completed"
    and _RCD_MORPH_TERMINAL.get("run_id")
    == "competition-real-handcrafted-division-gate-v1"
    and _RCD_MORPH_TERMINAL.get("feature_count") == 132
    and _RCD_MORPH_TERMINAL.get("competition_test_data_read") is False
    and _RCD_MORPH_TERMINAL.get("public_leaderboard_used_for_selection") is False
):
    raise RuntimeError("Strong-member source evidence is ineligible")

if _rcd_sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Pinned morphology runtime requires CPython 3.12, saw {_rcd_sys.version}")
_RCD_SKLEARN_ROOT = Path("/kaggle/working/strong-member-sklearn-1.9.0")
_rcd_subprocess.run(
    [
        _rcd_sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--disable-pip-version-check",
        "--quiet",
        "--target",
        str(_RCD_SKLEARN_ROOT),
        str(
            _RCD_ROOT
            / "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        ),
    ],
    check=True,
)
_rcd_sys.path.insert(0, str(_RCD_SKLEARN_ROOT))
_rcd_sys.path.insert(0, str(_RCD_ROOT))
import joblib as _rcd_joblib
import sklearn as _rcd_sklearn
import torch as _rcd_torch
from handcrafted_division import patch_features as _rcd_patch_features
from learned_division_recovery import (
    apply_ranked_consensus_division_recovery as _apply_ranked_consensus,
    discover_division_recovery_candidates as _discover_division_candidates,
)
from multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT as _RCD_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel as _RCDDeepModel,
)
from patch_model import sample_physical_patches as _rcd_sample_physical_patches

if _rcd_sklearn.__version__ != "1.9.0":
    raise RuntimeError(f"Pinned morphology sklearn changed: {_rcd_sklearn.__version__}")
_rcd_gpu_names = [
    _rcd_torch.cuda.get_device_name(index)
    for index in range(_rcd_torch.cuda.device_count())
]
if len(_rcd_gpu_names) != 2 or any("T4" not in name for name in _rcd_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_rcd_gpu_names}")

_RCD_DEVICE = _rcd_torch.device("cuda:0")
_RCD_DEEP_MODELS = []
for _member in _RCD_DEEP_MEMBERS:
    _model = _RCDDeepModel().to(_RCD_DEVICE)
    _model.load_state_dict(
        _rcd_torch.load(
            _RCD_ROOT / _member["path"],
            map_location=_RCD_DEVICE,
            weights_only=True,
        ),
        strict=True,
    )
    if sum(parameter.numel() for parameter in _model.parameters()) != _RCD_PARAMETER_COUNT:
        raise RuntimeError("Strong-member deep parameter inventory changed")
    _RCD_DEEP_MODELS.append(_model.requires_grad_(False).eval())

_RCD_MORPH_PAYLOAD = _rcd_joblib.load(
    _RCD_ROOT / "morphology_division_model.joblib"
)
if not (
    _RCD_MORPH_PAYLOAD.get("run_id")
    == "competition-real-handcrafted-division-gate-v1"
    and _RCD_MORPH_PAYLOAD.get("feature_family")
    == "temporal_radial_peak_morphology_v1"
    and _RCD_MORPH_PAYLOAD.get("feature_count") == 132
    and len(_RCD_MORPH_PAYLOAD.get("models", ())) == 2
):
    raise RuntimeError("Strong-member morphology payload changed")

_RCD_GEOMETRY_MINIMUM = float(_RCD_POLICY["biological_geometry_minimum"])
print({
    "strong_member_root": str(_RCD_ROOT),
    "deep_member_count": len(_RCD_DEEP_MODELS),
    "deep_parameters_per_member": _RCD_PARAMETER_COUNT,
    "deep_policy": _RCD_POLICY["deep_policy"],
    "base_safe_divisions_enabled": os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS") != "0",
    "absolute_threshold_used": False,
    "gpu_names": _rcd_gpu_names,
})
'''


RANKED_HELPERS = r'''
@_rcd_torch.inference_mode()
def _ranked_consensus_scores_for_candidates(nodes_by_id, edges, dataset):
    candidates = _discover_division_candidates(nodes_by_id, edges)
    parent_ids = sorted({
        candidate.parent_id
        for candidate in candidates
        if candidate.biological_geometry_score >= _RCD_GEOMETRY_MINIMUM
    })
    if not parent_ids:
        return {}, {}, {
            "candidate_parents_scored": 0,
            "candidate_frames_scored": 0,
            "deep_member_count": len(_RCD_DEEP_MODELS),
        }
    by_time = {}
    for parent_id in parent_ids:
        by_time.setdefault(int(nodes_by_id[parent_id]["t"]), []).append(parent_id)
    maximum_time = max(int(node["t"]) for node in nodes_by_id.values())
    frame_cache = {}
    deep_scores_by_model = [dict() for _model in _RCD_DEEP_MODELS]
    morphology_scores = {}
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
        patches = _rcd_sample_physical_patches(
            _rcd_torch.as_tensor(temporal, device=_RCD_DEVICE),
            centers,
            voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
            chunk_size=32,
        )
        frame_model_scores = []
        for _model in _RCD_DEEP_MODELS:
            parts = []
            for start in range(0, len(patches), 32):
                with _rcd_torch.autocast(
                    device_type="cuda", dtype=_rcd_torch.float16
                ):
                    _embedding, logits = _model(patches[start : start + 32])
                parts.append(logits.float().cpu())
            frame_model_scores.append(_rcd_torch.cat(parts).numpy())
        features = _rcd_patch_features(patches.float().cpu().numpy())
        morphology = np.mean(
            np.stack(
                [
                    row["model"].predict_proba(features)[:, 1]
                    for row in _RCD_MORPH_PAYLOAD["models"]
                ],
                axis=0,
            ),
            axis=0,
        )
        for local_index, parent_id in enumerate(frame_parent_ids):
            for member_index, values in enumerate(frame_model_scores):
                deep_scores_by_model[member_index][parent_id] = float(
                    values[local_index]
                )
            morphology_scores[parent_id] = float(morphology[local_index])
        del patches, features, frame_model_scores
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint - 1:
                del frame_cache[cached_timepoint]

    deep_rank_matrix = np.empty(
        (len(_RCD_DEEP_MODELS), len(parent_ids)), dtype=np.float64
    )
    for member_index, scores in enumerate(deep_scores_by_model):
        raw = np.asarray([scores[parent_id] for parent_id in parent_ids])
        order = np.argsort(-raw, kind="stable")
        percentiles = (
            np.ones(1, dtype=np.float64)
            if len(parent_ids) == 1
            else np.linspace(1.0, 0.0, len(parent_ids), dtype=np.float64)
        )
        deep_rank_matrix[member_index, order] = percentiles
    deep_consensus = deep_rank_matrix.mean(axis=0)
    deep_scores = {
        parent_id: float(deep_consensus[index])
        for index, parent_id in enumerate(parent_ids)
    }
    return deep_scores, morphology_scores, {
        "candidate_parents_scored": len(parent_ids),
        "candidate_frames_scored": len(by_time),
        "deep_member_count": len(_RCD_DEEP_MODELS),
    }


def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    deep_scores, morphology_scores, inference_stats = (
        _ranked_consensus_scores_for_candidates(nodes_by_id, edges, dataset)
    )
    recovered, recovery_stats = _apply_ranked_consensus(
        nodes_by_id,
        edges,
        deep_scores,
        morphology_scores,
        biological_geometry_minimum=_RCD_GEOMETRY_MINIMUM,
    )
    recovery_stats["candidate_parents_scored"] = inference_stats[
        "candidate_parents_scored"
    ]
    return recovered, {**inference_stats, **recovery_stats}

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
_evidence_div_denominator = _evidence_div_tp + _evidence_div_fp + _evidence_div_fn
_evidence_division = (
    _evidence_div_tp / _evidence_div_denominator
    if _evidence_div_denominator
    else 0.0
)
_evidence = {
    "schema_version": 1,
    "status": "completed_pending_external_promotion_gate",
    "run_id": "ema-strong-member-consensus-candidate-v2",
    "target_public_score": 0.945,
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "deep_policy": _RCD_POLICY["deep_policy"],
    "deep_member_count": len(_RCD_DEEP_MODELS),
    "deep_parameter_count_per_member": int(_RCD_PARAMETER_COUNT),
    "deep_model_sha256": [row["model_sha256"] for row in _RCD_DEEP_MEMBERS],
    "morphology_model_sha256": _RCD_POLICY["morphology_model_sha256"],
    "absolute_threshold_used": False,
    "runtime_manifest_sha256": _RCD_MANIFEST_SHA256,
    "base_safe_division_heuristic_enabled": (
        os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS") != "0"
        and _RCD_POLICY["base_safe_division_heuristic_enabled"] is True
    ),
    "external_policy_additive_only": _RCD_POLICY["external_policy_additive_only"],
    "ranked_geometric_candidates": int(
        _evidence_stats["ranked_consensus_geometric_candidates"].sum()
    ),
    "ranked_geometry_eligible_candidates": int(
        _evidence_stats["ranked_consensus_geometry_eligible_candidates"].sum()
    ),
    "ranked_candidate_parents_scored": int(
        _evidence_stats["ranked_consensus_candidate_parents_scored"].sum()
    ),
    "ranked_agreements": int(
        _evidence_stats["ranked_consensus_ranking_agreed"].sum()
    ),
    "ranked_edges_added": int(
        _evidence_stats["ranked_consensus_added_edges"].sum()
    ),
    "ranked_reassignments": int(
        _evidence_stats["ranked_consensus_reassignment_performed"].sum()
    ),
    "ranked_node_or_coordinate_changes": int(
        _evidence_stats["ranked_consensus_node_or_coordinate_changes"].sum()
    ),
    "validator_adjusted_edge_jaccard": _evidence_adjusted_edge,
    "validator_division_tp": _evidence_div_tp,
    "validator_division_fp": _evidence_div_fp,
    "validator_division_fn": _evidence_div_fn,
    "validator_division_jaccard": _evidence_division,
    "validator_proxy_score": _evidence_adjusted_edge + 0.10 * _evidence_division,
    "submission_sha256": _rcd_sha256(_evidence_submission),
    "competition_submission_performed": False,
    "authorized_for_submission": False,
}
_evidence_path = Path("/kaggle/working/candidate_evidence.json")
_evidence_path.write_text(
    _rcd_json.dumps(_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_rcd_json.dumps(_evidence, indent=2, sort_keys=True))
'''


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
        "title": "Biohub EMA Strong Member Candidate v2",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [
            "gpu",
            "cell-tracking",
            "strong-member-ensemble",
            "non-replica",
        ],
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
