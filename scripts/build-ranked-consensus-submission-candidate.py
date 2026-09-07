#!/usr/bin/env python
"""Build the attributed EMA plus ranked-consensus division candidate."""

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
TARGET_ID = "biohub-ema-ranked-consensus-candidate-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
CONSENSUS_REF = "indarkarhana/biohub-ranked-consensus-division-v1"


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


ATTRIBUTION = """## Project candidate: EMA control plus ranked division consensus

The detector, dual-seed ensemble, harmonic association, DeepCenter gap veto,
and constrained graph reconstruction are attributed to the public 0.927
lineage reproduced by `indarkarhana/biohub-clean-0-927-reproduction-v1`.
The running velocity EMA is independently reproduced from the documented idea
in `grafael/biohub-ct-0940-ema`.

The division stage is project-authored and was selected without leaderboard
feedback. Its first voter is a 46.4M-parameter temporal multiscale network
trained on real Biohub training patches on the Antelume A10G. Its second voter
is an independently trained 132-feature temporal morphology ensemble. Both
rank safe division recoveries strongly on the held-out development probe. An
edge is added only when both scale-invariant rankings choose the same
geometry-eligible parent, with at most one added edge per movie. No absolute
score threshold is used. The public rule-based safe-division adder is disabled.
"""


MODEL_SETUP_TEMPLATE = r'''# Strictly load the project ranked-consensus runtime and both frozen voters.
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
for _path in _RCD_INPUT_ROOT.rglob("RANKED_CONSENSUS_MANIFEST.json"):
    if _rcd_sha256(_path) != _RCD_MANIFEST_SHA256:
        continue
    _payload = _rcd_json.loads(_path.read_text(encoding="utf-8"))
    if _payload.get("run_id") == "competition-ranked-consensus-division-dataset-v1":
        _rcd_matches.append(_path.parent)
if len(_rcd_matches) != 1:
    raise RuntimeError(f"Expected one ranked-consensus runtime, saw {_rcd_matches}")
_RCD_ROOT = _rcd_matches[0]
_RCD_MANIFEST = _rcd_json.loads(
    (_RCD_ROOT / "RANKED_CONSENSUS_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _RCD_MANIFEST["files"].items():
    if _rcd_sha256(_RCD_ROOT / _name) != _record["sha256"]:
        raise RuntimeError(f"Ranked-consensus runtime changed: {_name}")

_RCD_POLICY = _rcd_json.loads(
    (_RCD_ROOT / "ranked-consensus-policy.json").read_text(encoding="utf-8")
)
_RCD_DEVELOPMENT = _rcd_json.loads(
    (_RCD_ROOT / "development_evidence.json").read_text(encoding="utf-8")
)
_RCD_DEEP_TERMINAL = _rcd_json.loads(
    (_RCD_ROOT / "deep_training_terminal.json").read_text(encoding="utf-8")
)
_RCD_MORPH_TERMINAL = _rcd_json.loads(
    (_RCD_ROOT / "morphology_training_terminal.json").read_text(encoding="utf-8")
)
if not (
    _RCD_POLICY.get("status") == "development_accepted"
    and _RCD_POLICY.get("run_id")
    == "competition-ranked-consensus-division-policy-v1"
    and _RCD_POLICY.get("absolute_threshold_used") is False
    and _RCD_POLICY.get("authorized_for_full_candidate_evaluation") is True
    and _RCD_POLICY.get("authorized_for_submission") is False
    and _RCD_POLICY.get("maximum_added_edges_per_movie") == 1
    and float(_RCD_POLICY.get("biological_geometry_minimum")) == 3.0
    and _RCD_POLICY.get("deep_model_sha256")
    == _rcd_sha256(_RCD_ROOT / "deep_division_model.pt")
    and _RCD_POLICY.get("morphology_model_sha256")
    == _rcd_sha256(_RCD_ROOT / "morphology_division_model.joblib")
    and _RCD_POLICY.get("sklearn_version") == "1.9.0"
    and _RCD_POLICY.get("sklearn_wheel_sha256")
    == _rcd_sha256(
        _RCD_ROOT
        / "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    and _RCD_DEVELOPMENT.get("status") == "development_positive"
    and _RCD_DEVELOPMENT.get("selected") == 3
    and _RCD_DEVELOPMENT.get("tp") == 3
    and _RCD_DEVELOPMENT.get("fp") == 0
    and _RCD_DEVELOPMENT.get("absolute_threshold_used") is False
    and _RCD_DEVELOPMENT.get("authorized_for_full_candidate_evaluation") is True
    and _RCD_DEVELOPMENT.get("authorized_for_submission") is False
    and _RCD_DEEP_TERMINAL.get("status") == "accepted_at_selection"
    and _RCD_DEEP_TERMINAL.get("run_id")
    == "competition-real-division-gate-v1"
    and _RCD_DEEP_TERMINAL.get("selection_gate_passed") is True
    and _RCD_DEEP_TERMINAL.get("competition_test_data_read") is False
    and _RCD_DEEP_TERMINAL.get("public_leaderboard_used_for_selection") is False
    and _RCD_DEEP_TERMINAL.get("ensemble_weights")
    == {"target_44b6": 0.0, "target_6bba": 1.0}
    and _RCD_MORPH_TERMINAL.get("status") == "completed"
    and _RCD_MORPH_TERMINAL.get("run_id")
    == "competition-real-handcrafted-division-gate-v1"
    and _RCD_MORPH_TERMINAL.get("feature_count") == 132
    and _RCD_MORPH_TERMINAL.get("competition_test_data_read") is False
    and _RCD_MORPH_TERMINAL.get("public_leaderboard_used_for_selection") is False
):
    raise RuntimeError("Ranked-consensus source evidence is ineligible")

if _rcd_sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Pinned morphology runtime requires CPython 3.12, saw {_rcd_sys.version}")
_RCD_SKLEARN_ROOT = Path("/kaggle/working/ranked-consensus-sklearn-1.9.0")
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
    raise RuntimeError(
        f"Pinned morphology sklearn changed: {_rcd_sklearn.__version__}"
    )

_rcd_gpu_names = [
    _rcd_torch.cuda.get_device_name(index)
    for index in range(_rcd_torch.cuda.device_count())
]
if len(_rcd_gpu_names) != 2 or any("T4" not in name for name in _rcd_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_rcd_gpu_names}")

_RCD_DEVICE = _rcd_torch.device("cuda:0")
_RCD_DEEP_MODEL = _RCDDeepModel().to(_RCD_DEVICE)
_RCD_DEEP_MODEL.load_state_dict(
    _rcd_torch.load(
        _RCD_ROOT / "deep_division_model.pt",
        map_location=_RCD_DEVICE,
        weights_only=True,
    ),
    strict=True,
)
if sum(parameter.numel() for parameter in _RCD_DEEP_MODEL.parameters()) != _RCD_PARAMETER_COUNT:
    raise RuntimeError("Ranked-consensus deep parameter inventory changed")
_RCD_DEEP_MODEL.requires_grad_(False).eval()

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
    raise RuntimeError("Ranked-consensus morphology payload changed")

_RCD_GEOMETRY_MINIMUM = float(
    _RCD_POLICY["biological_geometry_minimum"]
)
print({
    "ranked_consensus_root": str(_RCD_ROOT),
    "deep_parameters": _RCD_PARAMETER_COUNT,
    "morphology_models": [
        row["name"] for row in _RCD_MORPH_PAYLOAD["models"]
    ],
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
        }
    by_time = {}
    for parent_id in parent_ids:
        by_time.setdefault(int(nodes_by_id[parent_id]["t"]), []).append(parent_id)
    maximum_time = max(int(node["t"]) for node in nodes_by_id.values())
    frame_cache = {}
    deep_scores = {}
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
        deep_parts = []
        for start in range(0, len(patches), 32):
            with _rcd_torch.autocast(device_type="cuda", dtype=_rcd_torch.float16):
                _embedding, logits = _RCD_DEEP_MODEL(patches[start : start + 32])
            deep_parts.append(logits.float().cpu())
        deep = _rcd_torch.cat(deep_parts).numpy()
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
        for parent_id, deep_score, morphology_score in zip(
            frame_parent_ids, deep, morphology, strict=True
        ):
            deep_scores[parent_id] = float(deep_score)
            morphology_scores[parent_id] = float(morphology_score)
        del patches, features
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint - 1:
                del frame_cache[cached_timepoint]
    return deep_scores, morphology_scores, {
        "candidate_parents_scored": len(parent_ids),
        "candidate_frames_scored": len(by_time),
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


RANKED_APPLY = r'''
    if dataset is None:
        raise RuntimeError("Ranked consensus requires an explicit dataset")
    edges, ranked_consensus_stats = _apply_external_ranked_consensus(
        nodes_by_id, edges, dataset
    )
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise RuntimeError(
                f"{dataset}: ranked consensus produced a dangling edge "
                f"{source_id}->{target_id}"
            )
        source_time = int(nodes_by_id[source_id]["t"])
        target_time = int(nodes_by_id[target_id]["t"])
        if target_time != source_time + 1:
            raise RuntimeError(
                f"{dataset}: ranked consensus produced a nonconsecutive edge "
                f"{source_id}@{source_time}->{target_id}@{target_time}"
            )
    for key, value in ranked_consensus_stats.items():
        stats[f"ranked_consensus_{key}"] = value
    print(
        f"  [{dataset}] ranked consensus: "
        f"candidates={ranked_consensus_stats['geometric_candidates']} "
        f"eligible={ranked_consensus_stats['geometry_eligible_candidates']} "
        f"agreement={ranked_consensus_stats['ranking_agreed']} "
        f"added={ranked_consensus_stats['added_edges']}"
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
_evidence_div_denominator = _evidence_div_tp + _evidence_div_fp + _evidence_div_fn
_evidence_division = (
    _evidence_div_tp / _evidence_div_denominator
    if _evidence_div_denominator
    else 0.0
)
_evidence = {
    "schema_version": 1,
    "status": "completed_pending_external_promotion_gate",
    "run_id": "ema-ranked-consensus-candidate-v1",
    "target_public_score": 0.945,
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "project_division_components": 2,
    "deep_parameter_count": int(_RCD_PARAMETER_COUNT),
    "absolute_threshold_used": False,
    "runtime_manifest_sha256": _RCD_MANIFEST_SHA256,
    "deep_model_sha256": _RCD_POLICY["deep_model_sha256"],
    "morphology_model_sha256": _RCD_POLICY["morphology_model_sha256"],
    "safe_division_heuristic_disabled": (
        os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS") == "0"
    ),
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


def transform_notebook(
    consensus_root: Path,
    *,
    dataset_builder_path: Path | None = None,
    watchdog_run_id: str = "ema-ranked-consensus-candidate-v1",
    keep_base_safe_divisions: bool = False,
    attribution: str = ATTRIBUTION,
    model_setup_template: str = MODEL_SETUP_TEMPLATE,
    ranked_helpers: str = RANKED_HELPERS,
    candidate_evidence: str = CANDIDATE_EVIDENCE,
) -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Attributed public-control notebook changed")
    dataset_builder = runpy.run_path(
        str(
            dataset_builder_path
            or ROOT / "scripts/build-ranked-consensus-division-dataset.py"
        )
    )
    verified = dataset_builder["verify_dataset"](consensus_root)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="ascii"))

    watchdog = "".join(notebook["cells"][0]["source"])
    watchdog = replace_exact(
        watchdog,
        "_BIOHUB_RUN_ID = 'public-0927-clean-repro-v2'",
        f"_BIOHUB_RUN_ID = {watchdog_run_id!r}",
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
        f'os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "{int(keep_base_safe_divisions)}"',
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
        f'    "BIOHUB_OUTPUT_SAFE_DIVISIONS": "{int(keep_base_safe_divisions)}",',
    )
    guard = guard.replace(
        "Single model-level change: harmonic mutual-support association fusion",
        "Candidate changes: EMA motion plus ranked division consensus",
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
        ranked_helpers + "def filter_output_graph(\n",
    )
    post = replace_exact(
        post,
        "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
        RANKED_APPLY
        + "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
    )
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)

    inference_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Fail fast instead of silently running volumetric inference on CPU"
        in "".join(cell.get("source", []))
    )
    setup = model_setup_template.replace(
        "__MANIFEST_SHA256__", str(verified["manifest_sha256"])
    )
    notebook["cells"][inference_index + 1 : inference_index + 1] = [
        markdown_cell(attribution),
        code_cell(setup),
    ]
    terminal_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if '_biohub_write_terminal("completed")'
        in "".join(cell.get("source", []))
    )
    notebook["cells"][terminal_index:terminal_index] = [code_cell(candidate_evidence)]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-root", type=Path, required=True)
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
    notebook = transform_notebook(args.consensus_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub EMA Ranked Consensus Candidate v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [
            "gpu",
            "cell-tracking",
            "ranked-consensus",
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
