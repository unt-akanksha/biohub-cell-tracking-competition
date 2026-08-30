#!/usr/bin/env python
"""Build the additive EMA plus audit-passing relational consensus candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py"))
STRONG = runpy.run_path(str(ROOT / "scripts/build-strong-member-consensus-submission-candidate.py"))
DATASET_BUILDER = ROOT / "scripts/build-relational-consensus-division-dataset.py"
SOURCE_DIR = ROOT / "kaggle/biohub-ct-0940-ema"
SOURCE_METADATA_SHA256 = COMMON["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-ema-relational-consensus-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
CONSENSUS_REF = "indarkarhana/biohub-relational-consensus-division-v1"
RUN_ID = "ema-relational-consensus-candidate-v1"


ATTRIBUTION = """## Project candidate: EMA control plus heavy relational consensus

The detection, harmonic association control, safe-division rule, and graph
reconstruction retain their original public attribution. No public prediction
is copied. The additive division stage is project-authored and evaluates the
exact parent/retained-daughter/proposed-daughter tuple used during official-
train simulation.

Only checkpoints that passed movie-disjoint selection and sealed audit
independently are attached. Ensemble membership was frozen before audit.
Within-movie ranks are averaged without an absolute threshold, and a proposed
edge must be the identical top geometry-eligible candidate under the heavy
relational voter and an independent 132-feature temporal morphology voter.
At most one parent-free edge is added per movie. The notebook requires two T4
GPUs and partitions accepted relational members across them when at least two
independently strong members were admitted.
"""


MODEL_SETUP_TEMPLATE = r'''# Strictly load the additive relational-consensus runtime.
import concurrent.futures as _rcd_futures
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
for _path in _RCD_INPUT_ROOT.rglob("RELATIONAL_CONSENSUS_MANIFEST.json"):
    if _rcd_sha256(_path) != _RCD_MANIFEST_SHA256:
        continue
    _payload = _rcd_json.loads(_path.read_text(encoding="utf-8"))
    if _payload.get("run_id") == "competition-relational-consensus-division-dataset-v1":
        _rcd_matches.append(_path.parent)
if len(_rcd_matches) != 1:
    raise RuntimeError(f"Expected one relational runtime, saw {_rcd_matches}")
_RCD_ROOT = _rcd_matches[0]
_RCD_MANIFEST = _rcd_json.loads(
    (_RCD_ROOT / "RELATIONAL_CONSENSUS_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _RCD_MANIFEST["files"].items():
    if _rcd_sha256(_RCD_ROOT / _name) != _record["sha256"]:
        raise RuntimeError(f"Relational runtime changed: {_name}")

_RCD_POLICY = _rcd_json.loads(
    (_RCD_ROOT / "relational-consensus-policy.json").read_text(encoding="utf-8")
)
_RCD_SELECTION = _rcd_json.loads(
    (_RCD_ROOT / "selection_audit_terminal.json").read_text(encoding="utf-8")
)
_RCD_PROBE = _rcd_json.loads(
    (_RCD_ROOT / "relational_development_probe.json").read_text(encoding="utf-8")
)
_RCD_DEVELOPMENT = _rcd_json.loads(
    (_RCD_ROOT / "relational_development_evidence.json").read_text(encoding="utf-8")
)
_RCD_MORPH_TERMINAL = _rcd_json.loads(
    (_RCD_ROOT / "morphology_training_terminal.json").read_text(encoding="utf-8")
)
_RCD_DEEP_MEMBERS = _RCD_POLICY.get("relational_members", [])
if not (
    _RCD_POLICY.get("schema_version") == 1
    and _RCD_POLICY.get("status") == "development_accepted"
    and _RCD_POLICY.get("run_id") == "competition-relational-consensus-division-policy-v1"
    and _RCD_POLICY.get("relational_policy")
    in {"equal_rank_selection_admitted_ensemble", "strongest_selection_individual"}
    and 1 <= len(_RCD_DEEP_MEMBERS) <= 8
    and _RCD_POLICY.get("relational_member_count") == len(_RCD_DEEP_MEMBERS)
    and len({row["model_sha256"] for row in _RCD_DEEP_MEMBERS}) == len(_RCD_DEEP_MEMBERS)
    and all(
        row.get("model_sha256") == _rcd_sha256(_RCD_ROOT / row["path"])
        and row.get("parameter_count") == 48_313_050
        and row.get("selection_gate_passed") is True
        and row.get("audit_gate_passed") is True
        for row in _RCD_DEEP_MEMBERS
    )
    and _RCD_POLICY.get("base_safe_division_heuristic_enabled") is True
    and _RCD_POLICY.get("external_policy_additive_only") is True
    and _RCD_POLICY.get("exact_two_t4_required") is True
    and _RCD_POLICY.get("absolute_threshold_used") is False
    and _RCD_POLICY.get("model_subset_searched_on_audit") is False
    and _RCD_POLICY.get("maximum_added_edges_per_movie") == 1
    and float(_RCD_POLICY.get("biological_geometry_minimum")) == 3.0
    and _RCD_POLICY.get("morphology_model_sha256")
    == _rcd_sha256(_RCD_ROOT / "morphology_division_model.joblib")
    and _RCD_SELECTION.get("policy_audit_passed") is True
    and _RCD_SELECTION.get("deployment_policy") == _RCD_POLICY.get("relational_policy")
    and _RCD_SELECTION.get("deployment_members")
    == [row["member"] for row in _RCD_DEEP_MEMBERS]
    and _RCD_SELECTION.get("absolute_threshold_used_for_deployment") is False
    and _RCD_SELECTION.get("model_subset_searched_on_audit") is False
    and _RCD_SELECTION.get("final_probe_opened") is False
    and _RCD_PROBE.get("status") == "development_probe_complete"
    and _RCD_PROBE.get("member_count") == len(_RCD_DEEP_MEMBERS)
    and _RCD_PROBE.get("absolute_threshold_used") is False
    and _RCD_PROBE.get("weights_searched_on_probe") is False
    and _RCD_PROBE.get("model_subset_searched_on_probe") is False
    and _RCD_DEVELOPMENT.get("status") == "development_positive"
    and _RCD_DEVELOPMENT.get("selected") == 3
    and _RCD_DEVELOPMENT.get("tp") == 3
    and _RCD_DEVELOPMENT.get("fp") == 0
    and _RCD_DEVELOPMENT.get("relational_probe_sha256")
    == _rcd_sha256(_RCD_ROOT / "relational_development_probe.json")
    and _RCD_DEVELOPMENT.get("authorized_for_full_candidate_evaluation") is True
    and _RCD_DEVELOPMENT.get("authorized_for_submission") is False
    and _RCD_MORPH_TERMINAL.get("status") == "completed"
    and _RCD_MORPH_TERMINAL.get("run_id") == "competition-real-handcrafted-division-gate-v1"
    and _RCD_MORPH_TERMINAL.get("feature_count") == 132
    and _RCD_MORPH_TERMINAL.get("competition_test_data_read") is False
    and _RCD_MORPH_TERMINAL.get("public_leaderboard_used_for_selection") is False
):
    raise RuntimeError("Relational source evidence is ineligible")

if _rcd_sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Pinned morphology runtime requires CPython 3.12, saw {_rcd_sys.version}")
_RCD_SKLEARN_ROOT = Path("/kaggle/working/relational-sklearn-1.9.0")
_rcd_subprocess.run(
    [
        _rcd_sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--disable-pip-version-check", "--quiet", "--target", str(_RCD_SKLEARN_ROOT),
        str(_RCD_ROOT / "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
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
from patch_model import sample_physical_patches as _rcd_sample_physical_patches
from relational_division_inference import score_relational_candidates as _rcd_score_relational_candidates
from relational_division_model import RelationalDivisionModel as _RCDDeepModel

if _rcd_sklearn.__version__ != "1.9.0":
    raise RuntimeError(f"Pinned morphology sklearn changed: {_rcd_sklearn.__version__}")
_rcd_gpu_names = [_rcd_torch.cuda.get_device_name(index) for index in range(_rcd_torch.cuda.device_count())]
if len(_rcd_gpu_names) != 2 or any("T4" not in name for name in _rcd_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_rcd_gpu_names}")
_RCD_DEVICES = [_rcd_torch.device("cuda:0"), _rcd_torch.device("cuda:1")]
_RCD_MODELS_BY_DEVICE = [[], []]
_RCD_DEEP_MODELS = []
_RCD_PARAMETER_COUNT = 48_313_050
for _index, _member in enumerate(_RCD_DEEP_MEMBERS):
    _device_index = _index % 2
    _model = _RCDDeepModel().to(_RCD_DEVICES[_device_index])
    _model.load_state_dict(
        _rcd_torch.load(_RCD_ROOT / _member["path"], map_location=_RCD_DEVICES[_device_index], weights_only=True),
        strict=True,
    )
    if sum(parameter.numel() for parameter in _model.parameters()) != _RCD_PARAMETER_COUNT:
        raise RuntimeError("Relational parameter inventory changed")
    _model.requires_grad_(False).eval()
    _RCD_MODELS_BY_DEVICE[_device_index].append(_model)
    _RCD_DEEP_MODELS.append(_model)

_RCD_MORPH_PAYLOAD = _rcd_joblib.load(_RCD_ROOT / "morphology_division_model.joblib")
if not (
    _RCD_MORPH_PAYLOAD.get("run_id") == "competition-real-handcrafted-division-gate-v1"
    and _RCD_MORPH_PAYLOAD.get("feature_family") == "temporal_radial_peak_morphology_v1"
    and _RCD_MORPH_PAYLOAD.get("feature_count") == 132
    and len(_RCD_MORPH_PAYLOAD.get("models", ())) == 2
):
    raise RuntimeError("Relational morphology payload changed")
_RCD_DEVICE = _RCD_DEVICES[0]
_RCD_GEOMETRY_MINIMUM = 3.0
print({
    "relational_root": str(_RCD_ROOT),
    "relational_member_count": len(_RCD_DEEP_MODELS),
    "members_per_gpu": [len(row) for row in _RCD_MODELS_BY_DEVICE],
    "parameters_per_member": _RCD_PARAMETER_COUNT,
    "relational_policy": _RCD_POLICY["relational_policy"],
    "absolute_threshold_used": False,
    "gpu_names": _rcd_gpu_names,
})
'''


RANKED_HELPERS = r'''
def _relational_scores_for_candidates(nodes_by_id, edges, dataset):
    candidates = _discover_division_candidates(nodes_by_id, edges)
    eligible = [
        candidate for candidate in candidates
        if candidate.biological_geometry_score >= _RCD_GEOMETRY_MINIMUM
    ]
    if not eligible:
        return {}, {}, {
            "candidate_parents_scored": 0,
            "candidate_frames_scored": 0,
            "deep_member_count": len(_RCD_DEEP_MODELS),
            "gpu_groups_used": 0,
        }
    maximum_time = max(int(node["t"]) for node in nodes_by_id.values())

    def _score_device_group(device_index):
        models = _RCD_MODELS_BY_DEVICE[device_index]
        frame_cache = {}
        scores, stats = _rcd_score_relational_candidates(
            models,
            eligible,
            nodes_by_id,
            edges,
            read_frame=lambda index: read_test_frame(dataset, index, frame_cache),
            sample_physical_patches=_rcd_sample_physical_patches,
            device=_RCD_DEVICES[device_index],
            maximum_time=maximum_time,
            batch_size=12,
        )
        return len(models), scores, stats

    active_groups = [index for index, models in enumerate(_RCD_MODELS_BY_DEVICE) if models]
    with _rcd_futures.ThreadPoolExecutor(max_workers=len(active_groups)) as executor:
        group_results = list(executor.map(_score_device_group, active_groups))
    total_models = sum(row[0] for row in group_results)
    parent_ids = sorted(candidate.parent_id for candidate in eligible)
    relational_scores = {
        parent_id: float(
            sum(count * scores[parent_id] for count, scores, _stats in group_results)
            / total_models
        )
        for parent_id in parent_ids
    }

    by_time = {}
    for candidate in eligible:
        by_time.setdefault(int(nodes_by_id[candidate.parent_id]["t"]), []).append(candidate.parent_id)
    morphology_scores = {}
    frame_cache = {}
    for timepoint, frame_parent_ids in sorted(by_time.items()):
        frame_indices = [max(0, min(maximum_time, timepoint + offset)) for offset in (-1, 0, 1)]
        temporal = np.stack([read_test_frame(dataset, index, frame_cache) for index in frame_indices], axis=0)
        centers = np.asarray(
            [[nodes_by_id[parent_id]["z"], nodes_by_id[parent_id]["y"], nodes_by_id[parent_id]["x"]] for parent_id in frame_parent_ids],
            dtype=np.float32,
        )
        patches = _rcd_sample_physical_patches(
            _rcd_torch.as_tensor(temporal, device=_RCD_DEVICE),
            centers,
            voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
            chunk_size=32,
        )
        features = _rcd_patch_features(patches.float().cpu().numpy())
        morphology = np.mean(
            np.stack([row["model"].predict_proba(features)[:, 1] for row in _RCD_MORPH_PAYLOAD["models"]], axis=0),
            axis=0,
        )
        for parent_id, value in zip(frame_parent_ids, morphology, strict=True):
            morphology_scores[parent_id] = float(value)
        del patches, features
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint - 1:
                del frame_cache[cached_timepoint]
    return relational_scores, morphology_scores, {
        "candidate_parents_scored": len(parent_ids),
        "candidate_frames_scored": len(by_time),
        "deep_member_count": total_models,
        "gpu_groups_used": len(active_groups),
    }


def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    deep_scores, morphology_scores, inference_stats = _relational_scores_for_candidates(
        nodes_by_id, edges, dataset
    )
    recovered, recovery_stats = _apply_ranked_consensus(
        nodes_by_id,
        edges,
        deep_scores,
        morphology_scores,
        biological_geometry_minimum=_RCD_GEOMETRY_MINIMUM,
    )
    recovery_stats["candidate_parents_scored"] = inference_stats["candidate_parents_scored"]
    recovery_stats["gpu_groups_used"] = inference_stats["gpu_groups_used"]
    return recovered, {**inference_stats, **recovery_stats}

'''


CANDIDATE_EVIDENCE = (
    STRONG["CANDIDATE_EVIDENCE"]
    .replace("ema-strong-member-consensus-candidate-v2", RUN_ID)
    .replace('_RCD_POLICY["deep_policy"]', '_RCD_POLICY["relational_policy"]')
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
        "title": "Biohub EMA Relational Consensus v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "relational-division", "non-replica"],
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
