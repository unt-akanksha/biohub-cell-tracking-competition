#!/usr/bin/env python
"""Build the clean 948TTA2 plus frozen unanimous graph-salvage evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
V3 = runpy.run_path(str(ROOT / "scripts/build-948tta2-fresh-graph-candidate-v3.py"))
GRAPH = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-consensus-submission-candidate.py")
)
DEPLOY = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-unanimous-salvage-deployment-v4.py")
)
TARGET_ID = "biohub-948tta2-unanimous-salvage-v4"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-graph-context-unanimous-salvage-v4"
RUN_ID = "948tta2-unanimous-salvage-v4"
FROZEN_VALIDATION_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)


ATTRIBUTION = """## Clean public base plus project-authored unanimous graph salvage

The detector, association ensemble, edge-feature TTA, harmonic bidirectional
fusion, DeepCenter veto, reconstruction, and safe-division stage retain full
attribution to `redoctopusk/biohub-948tta2` and its declared Pilkwang inputs.
No public prediction is copied and leaderboard results are not selection data.

The additive project component is deliberately sparse. Four independently
initialized 74.7M-parameter graph-context models must name the identical top
geometry-eligible parent, every raw logit must exceed that member's pre-audit
zero-false-positive selection threshold, and an independent 132-feature
morphology ensemble must name the same parent. At most one edge is added per
movie. The policy and thresholds were frozen before these four cross-family,
complete validation movies were opened; none was used to train the graph
models. The run creates evidence, not a competition submission.
"""


MODEL_SETUP_HEADER = r'''# Strictly load the frozen evaluation-only salvage runtime.
import concurrent.futures as _gcd_futures
import copy as _gcd_copy
import hashlib as _gcd_hashlib
import json as _gcd_json
import subprocess as _gcd_subprocess
import sys as _gcd_sys

_GCD_MANIFEST_SHA256 = "__MANIFEST_SHA256__"
_GCD_INPUT_ROOT = Path("/kaggle/input")

def _gcd_sha256(path):
    digest = _gcd_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_gcd_matches = [
    path.parent
    for path in _GCD_INPUT_ROOT.rglob("GRAPH_CONTEXT_SALVAGE_MANIFEST.json")
    if _gcd_sha256(path) == _GCD_MANIFEST_SHA256
]
if len(_gcd_matches) != 1:
    raise RuntimeError(f"Expected one unanimous salvage runtime, saw {_gcd_matches}")
_GCD_ROOT = _gcd_matches[0]
_GCD_MANIFEST = _gcd_json.loads(
    (_GCD_ROOT / "GRAPH_CONTEXT_SALVAGE_MANIFEST.json").read_text(encoding="utf-8")
)
_GCD_POLICY = _gcd_json.loads(
    (_GCD_ROOT / "graph-context-unanimous-salvage-policy.json").read_text(encoding="utf-8")
)
for _name, _record in _GCD_MANIFEST["files"].items():
    _path = _GCD_ROOT / _name
    if _path.stat().st_size != _record["bytes"] or _gcd_sha256(_path) != _record["sha256"]:
        raise RuntimeError(f"Unanimous salvage runtime changed: {_name}")

_GCD_DEEP_MEMBERS = _GCD_POLICY.get("members", [])
_GCD_THRESHOLDS = [float(row["raw_logit_threshold"]) for row in _GCD_DEEP_MEMBERS]
if not (
    _GCD_MANIFEST.get("run_id") == "competition-graph-context-unanimous-salvage-deployment-v4"
    and _GCD_MANIFEST.get("status") == "evaluation_only"
    and _GCD_MANIFEST.get("submission_command_included") is False
    and _GCD_POLICY.get("run_id") == "competition-graph-context-unanimous-salvage-v4"
    and _GCD_POLICY.get("status") == "frozen_before_cross_family_movie_evaluation"
    and _GCD_POLICY.get("minimum_member_agreement") == 4
    and _GCD_POLICY.get("morphology_top_parent_agreement_required") is True
    and _GCD_POLICY.get("maximum_added_edges_per_movie") == 1
    and float(_GCD_POLICY.get("biological_geometry_minimum")) == 3.0
    and _GCD_POLICY.get("cross_family_validation_stems") == __FROZEN_STEMS__
    and _GCD_POLICY.get("cross_family_stems_seen_by_graph_training") is False
    and _GCD_POLICY.get("audit_scores_used_to_set_member_thresholds") is False
    and _GCD_POLICY.get("leaderboard_used_for_policy_selection") is False
    and _GCD_POLICY.get("metric_hack_used") is False
    and _GCD_POLICY.get("authorized_for_submission") is False
    and len(_GCD_DEEP_MEMBERS) == 4
    and all(
        row.get("parameter_count") == 74_732_308
        and row.get("selection_threshold_evidence", {}).get("fp") == 0
        and row.get("selection_threshold_evidence", {}).get("tp", 0) >= 2
        and row.get("raw_logit_threshold")
            == row.get("selection_threshold_evidence", {}).get("threshold")
        and row.get("model_sha256") == _gcd_sha256(_GCD_ROOT / row["path"])
        for row in _GCD_DEEP_MEMBERS
    )
):
    raise RuntimeError("Unanimous graph salvage evidence is ineligible")

'''


def model_setup(manifest_sha256: str) -> str:
    legacy = V3["model_setup"]("unused")
    tail = legacy[legacy.index("if _gcd_sys.version_info"):]
    tail = tail.replace(
        '"graph_context_policy": _GCD_POLICY["graph_context_policy"],\n'
        '    "absolute_threshold_used": False,',
        '"graph_context_policy": _GCD_POLICY["deployment_policy"],\n'
        '    "absolute_threshold_used": True,\n'
        '    "member_thresholds": _GCD_THRESHOLDS,',
    )
    header = MODEL_SETUP_HEADER.replace("__MANIFEST_SHA256__", manifest_sha256)
    header = header.replace("__FROZEN_STEMS__", repr(list(FROZEN_VALIDATION_STEMS)))
    return header + tail


def ranked_helpers() -> str:
    source = GRAPH["RANKED_HELPERS"]
    source = source.replace(
        '[row["model"].predict_proba(features)[:, 1] for row in _GCD_MORPH_PAYLOAD["models"]]',
        '[model.predict_proba(features)[:, 1] for model in _GCD_MORPH_PAYLOAD["models"]]',
    )
    source = source.replace(
        "    deep_scores = _gcd_calibration_free_parent_scores(raw_scores_by_model, parent_ids)\n"
        "    return deep_scores, morphology_scores, {",
        "    return raw_scores_by_model, morphology_scores, {",
    )
    old_apply = source[source.index("def _apply_external_ranked_consensus"):]
    new_apply = r'''def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    if os.environ.get("BIOHUB_UNANIMOUS_SALVAGE_ENABLE", "1") == "0":
        return edges, {
            "geometric_candidates": 0, "geometry_eligible_candidates": 0,
            "ranking_agreed": 0, "thresholds_passed": 0, "added_edges": 0,
            "reassignment_performed": 0, "node_or_coordinate_changes": 0,
            "candidate_parents_scored": 0, "candidate_frames_scored": 0,
            "deep_member_count": len(_GCD_DEEP_MODELS), "gpu_groups_used": 0,
        }
    raw_scores, morphology_scores, inference_stats = _graph_context_scores_for_candidates(
        nodes_by_id, edges, dataset
    )
    selected_parent = None
    unanimous = False
    thresholds_passed = False
    morphology_agreed = False
    if raw_scores and all(scores for scores in raw_scores) and morphology_scores:
        member_tops = [
            max(scores, key=lambda parent_id: (float(scores[parent_id]), -int(parent_id)))
            for scores in raw_scores
        ]
        unanimous = len(set(member_tops)) == 1
        if unanimous:
            candidate_parent = member_tops[0]
            thresholds_passed = all(
                float(scores[candidate_parent]) >= threshold
                for scores, threshold in zip(raw_scores, _GCD_THRESHOLDS, strict=True)
            )
            morphology_top = max(
                morphology_scores,
                key=lambda parent_id: (float(morphology_scores[parent_id]), -int(parent_id)),
            )
            morphology_agreed = morphology_top == candidate_parent
            if thresholds_passed and morphology_agreed:
                selected_parent = candidate_parent
    deep_gate = {} if selected_parent is None else {selected_parent: 1.0}
    morphology_gate = {} if selected_parent is None else {selected_parent: 1.0}
    recovered, recovery_stats = _apply_ranked_consensus(
        nodes_by_id, edges, deep_gate, morphology_gate,
        biological_geometry_minimum=_GCD_GEOMETRY_MINIMUM,
    )
    recovery_stats.update({
        "all_member_top_parent_agreed": int(unanimous),
        "all_member_thresholds_passed": int(thresholds_passed),
        "morphology_top_parent_agreed": int(morphology_agreed),
        "absolute_threshold_used": True,
        "candidate_parents_scored": inference_stats["candidate_parents_scored"],
        "candidate_frames_scored": inference_stats["candidate_frames_scored"],
        "deep_member_count": inference_stats["deep_member_count"],
        "gpu_groups_used": inference_stats["gpu_groups_used"],
    })
    return recovered, recovery_stats

'''
    return source.replace(old_apply, new_apply)


EVIDENCE = V3["EVIDENCE"].replace("fresh_graph", "unanimous_salvage")
EVIDENCE = EVIDENCE.replace("948tta2-fresh-graph-v3", RUN_ID)
EVIDENCE = EVIDENCE.replace(
    '"unanimous_salvage_policy_sha256": _gcd_sha256(\n'
    '        _GCD_ROOT / "graph-context-fresh-policy.json"\n'
    "    ),",
    '"unanimous_salvage_policy_sha256": _gcd_sha256(\n'
    '        _GCD_ROOT / "graph-context-unanimous-salvage-policy.json"\n'
    "    ),",
)
EVIDENCE = EVIDENCE.replace(
    '"graph_member_count": len(_GCD_DEEP_MEMBERS),',
    '"graph_member_count": len(_GCD_DEEP_MEMBERS),\n'
    '    "raw_member_thresholds": _GCD_THRESHOLDS,\n'
    '    "validation_stems_seen_by_graph_training": False,',
)
EVIDENCE = EVIDENCE.replace(
    '"morphology_model_sha256": _GCD_POLICY["morphology_model_sha256"],',
    '"morphology_model_sha256": _gcd_sha256(_GCD_ROOT / "morphology_model.joblib"),',
)


def build_notebook(graph_root: Path) -> dict:
    globals_map = V3["build_notebook"].__globals__
    overrides = {
        "DEPLOY": DEPLOY,
        "RUN_ID": RUN_ID,
        "FROZEN_VALIDATION_STEMS": FROZEN_VALIDATION_STEMS,
        "ATTRIBUTION": ATTRIBUTION,
        "model_setup": model_setup,
        "ranked_helpers": ranked_helpers,
        "EVIDENCE": EVIDENCE,
    }
    previous = {key: globals_map[key] for key in overrides}
    globals_map.update(overrides)
    try:
        notebook = V3["build_notebook"](graph_root)
    finally:
        globals_map.update(previous)
    serialized = json.dumps(notebook)
    serialized = serialized.replace("BIOHUB_FRESH_GRAPH_ENABLE", "BIOHUB_UNANIMOUS_SALVAGE_ENABLE")
    serialized = serialized.replace("fresh_graph", "unanimous_salvage")
    notebook = json.loads(serialized)
    notebook["metadata"]["codex"].update(
        {
            "run_id": RUN_ID,
            "frozen_validation_stems": list(FROZEN_VALIDATION_STEMS),
            "validation_stems_seen_by_graph_training": False,
            "submission_command_included": False,
        }
    )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    graph_root = args.graph_root.resolve()
    if TARGET_DIR.exists():
        if not args.replace:
            raise FileExistsError(TARGET_DIR)
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook(graph_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(V3["SOURCE_METADATA"].read_text(encoding="utf-8"))
    dataset_sources = list(base_metadata.get("dataset_sources", []))
    if RUNTIME_REF not in dataset_sources:
        dataset_sources.append(RUNTIME_REF)
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub 948TTA2 Unanimous Salvage v4",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "unanimous-salvage", "non-replica"],
        "dataset_sources": dataset_sources,
        "kernel_sources": list(base_metadata.get("kernel_sources", [])),
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata.pop("id_no", None)
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
