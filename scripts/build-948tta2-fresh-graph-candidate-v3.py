from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts/build-948tta2-lsm-consensus-candidate.py"))
GRAPH = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-consensus-submission-candidate.py")
)
DEPLOY = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-fresh-deployment-v3.py")
)
sha256_file = BASE["sha256_file"]
SOURCE_NOTEBOOK = BASE["SOURCE_NOTEBOOK"]
SOURCE_METADATA = BASE["SOURCE_METADATA"]
SOURCE_NOTEBOOK_SHA256 = BASE["SOURCE_NOTEBOOK_SHA256"]
SOURCE_METADATA_SHA256 = BASE["SOURCE_METADATA_SHA256"]
TARGET_ID = "biohub-948tta2-fresh-graph-v3"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-graph-context-fresh-consensus-v3"
RUN_ID = "948tta2-fresh-graph-v3"
FROZEN_VALIDATION_STEMS = (
    "44b6_d754aa59",
    "44b6_7a302da0",
    "6bba_debd7bfa",
    "6bba_fc5f39dc",
)


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


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"948TTA2 graph candidate drifted for {old!r}: expected {count}, saw {actual}"
        )
    return text.replace(old, new, count)


def find_cell(notebook: dict, pattern: str) -> int:
    matches = [
        index
        for index, cell in enumerate(notebook["cells"])
        if pattern in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"948TTA2 graph cell drift for {pattern!r}: {matches}")
    return matches[0]


ATTRIBUTION = """## Clean public base plus project-authored graph-context ensemble

The detector, dual-seed association ensemble, edge-feature TTA, harmonic
bidirectional fusion, DeepCenter veto, reconstruction, and safe-division stage
retain attribution to `redoctopusk/biohub-948tta2` and its declared Pilkwang
inputs. Its public leaderboard result is context only, not selection evidence,
and no public prediction is copied.

The additive component is project-authored: all fresh-split-admitted 74.7M
graph-context models are equal-rank ensembled and must select the identical
geometry-eligible parent as an independently fitted 132-feature morphology
ensemble. At most one parent-free edge is added per movie. Membership was
frozen on a hash split before sealed audit, with no absolute score threshold,
model weighting, audit subset search, or leaderboard feedback.

The candidate and untouched control are scored on the same four predeclared
complete movies using the patched metric proxy. Submission eligibility
requires positive pooled gain, zero per-movie adjusted-edge regression,
strict division gain, at least one production edge, and graph integrity.
"""


MODEL_SETUP_HEADER = r'''# Strictly load the fresh-audit graph deployment runtime.
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
    for path in _GCD_INPUT_ROOT.rglob("GRAPH_CONTEXT_FRESH_MANIFEST.json")
    if _gcd_sha256(path) == _GCD_MANIFEST_SHA256
]
if len(_gcd_matches) != 1:
    raise RuntimeError(f"Expected one fresh graph runtime, saw {_gcd_matches}")
_GCD_ROOT = _gcd_matches[0]
_GCD_MANIFEST = _gcd_json.loads(
    (_GCD_ROOT / "GRAPH_CONTEXT_FRESH_MANIFEST.json").read_text(encoding="utf-8")
)
for _name, _record in _GCD_MANIFEST["files"].items():
    _path = _GCD_ROOT / _name
    if _path.stat().st_size != _record["bytes"] or _gcd_sha256(_path) != _record["sha256"]:
        raise RuntimeError(f"Fresh graph runtime changed: {_name}")

_GCD_POLICY = _gcd_json.loads(
    (_GCD_ROOT / "graph-context-fresh-policy.json").read_text(encoding="utf-8")
)
_GCD_SELECTION = _gcd_json.loads(
    (_GCD_ROOT / "fresh_training_terminal.json").read_text(encoding="utf-8")
)
_GCD_VERIFICATION = _gcd_json.loads(
    (_GCD_ROOT / "fresh_verification_report.json").read_text(encoding="utf-8")
)
_GCD_DEEP_MEMBERS = _GCD_POLICY.get("graph_context_members", [])
if not (
    _GCD_MANIFEST.get("run_id") == "competition-graph-context-fresh-deployment-v3"
    and _GCD_POLICY.get("status") == "fresh_audit_accepted"
    and _GCD_POLICY.get("run_id") == "competition-graph-context-fresh-policy-v3"
    and _GCD_POLICY.get("graph_context_policy")
        == "all-selection-admitted-equal-rank-ensemble"
    and 2 <= len(_GCD_DEEP_MEMBERS) <= 4
    and _GCD_POLICY.get("graph_context_member_count") == len(_GCD_DEEP_MEMBERS)
    and len({row["model_sha256"] for row in _GCD_DEEP_MEMBERS}) == len(_GCD_DEEP_MEMBERS)
    and all(
        row.get("model_sha256") == _gcd_sha256(_GCD_ROOT / row["path"])
        and row.get("parameter_count") == 74_732_308
        and row.get("selection_gate_passed") is True
        for row in _GCD_DEEP_MEMBERS
    )
    and _GCD_POLICY.get("fresh_audit_gate_passed") is True
    and _GCD_POLICY.get("selection_policy_frozen_before_audit") is True
    and _GCD_POLICY.get("absolute_threshold_used") is False
    and _GCD_POLICY.get("model_weights_searched_on_audit") is False
    and _GCD_POLICY.get("model_subset_searched_on_audit") is False
    and _GCD_POLICY.get("maximum_added_edges_per_movie") == 1
    and float(_GCD_POLICY.get("biological_geometry_minimum")) == 3.0
    and _GCD_SELECTION.get("policy_audit_passed") is True
    and _GCD_SELECTION.get("selection_policy_frozen_before_audit") is True
    and _GCD_SELECTION.get("audit_opened_after_policy_freeze") is True
    and _GCD_SELECTION.get("competition_test_data_read") is False
    and _GCD_SELECTION.get("public_predictions_copied") is False
    and _GCD_SELECTION.get("public_leaderboard_used_for_selection") is False
    and _GCD_VERIFICATION.get("status") == "verified_fresh_audit_acceptance"
    and _GCD_VERIFICATION.get("deployment_members")
        == [row["member"] for row in _GCD_DEEP_MEMBERS]
    and _GCD_VERIFICATION.get("authorized_for_full_candidate_evaluation") is True
    and _GCD_VERIFICATION.get("authorized_for_submission") is False
):
    raise RuntimeError("Fresh graph source evidence is ineligible")

'''


def model_setup(manifest_sha256: str) -> str:
    legacy = GRAPH["MODEL_SETUP_TEMPLATE"]
    tail = legacy[legacy.index("if _gcd_sys.version_info"):]
    tail = tail.replace(
        "morphology_division_model.joblib", "morphology_model.joblib"
    ).replace(
        '== "competition-real-handcrafted-division-gate-v1"',
        '== "competition-graph-context-fresh-ensemble-v3"',
    )
    return MODEL_SETUP_HEADER.replace("__MANIFEST_SHA256__", manifest_sha256) + tail


def ranked_helpers() -> str:
    source = GRAPH["RANKED_HELPERS"]
    marker = "def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):\n"
    guard = r'''def _apply_external_ranked_consensus(nodes_by_id, edges, dataset):
    if os.environ.get("BIOHUB_FRESH_GRAPH_ENABLE", "1") == "0":
        return edges, {
            "geometric_candidates": 0,
            "geometry_eligible_candidates": 0,
            "ranking_agreed": 0,
            "added_edges": 0,
            "reassignment_performed": 0,
            "node_or_coordinate_changes": 0,
            "candidate_parents_scored": 0,
            "candidate_frames_scored": 0,
            "deep_member_count": len(_GCD_DEEP_MODELS),
            "gpu_groups_used": 0,
        }
'''
    return replace_exact(source, marker, guard)


OFFLOAD_MODELS = r'''# Park graph models while the public validator predicts.
for _gcd_group in _GCD_MODELS_BY_DEVICE:
    for _gcd_index, _gcd_model in _gcd_group:
        _gcd_model.to("cpu")
for _gcd_device in _GCD_DEVICES:
    with _gcd_torch.cuda.device(_gcd_device):
        _gcd_torch.cuda.empty_cache()
print("Fresh graph models parked on CPU for validator inference.")
'''


RESTORE_MODELS = r'''# Restore the immutable graph ensemble for paired scoring.
for _gcd_device_index, _gcd_group in enumerate(_GCD_MODELS_BY_DEVICE):
    for _gcd_index, _gcd_model in _gcd_group:
        _gcd_model.to(_GCD_DEVICES[_gcd_device_index])
print("Fresh graph models restored for complete-movie scoring.")
'''


VALIDATOR_LOOP_NEW = r'''    validator_rows_by_arm = {"control": [], "fresh_graph": []}
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        pred_path = val_pred_paths.get(stem)
        if not gt_path.exists() or pred_path is None:
            print(f"VALIDATOR: skipping {stem} (missing GT or prediction .geff)")
            continue
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)

        pred_graph = graph_from_geff(pred_path)
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        _real_test_dir = TEST_DIR
        _real_graph_flag = os.environ.get("BIOHUB_FRESH_GRAPH_ENABLE")
        globals()["TEST_DIR"] = TRAIN_DIR
        try:
            os.environ["BIOHUB_FRESH_GRAPH_ENABLE"] = "0"
            control_nodes, control_edges, _control_stats = filter_output_graph(
                raw_nodes_by_id, raw_edges, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            candidate_nodes = _gcd_copy.deepcopy(control_nodes)
            candidate_edges = _gcd_copy.deepcopy(control_edges)
            os.environ["BIOHUB_FRESH_GRAPH_ENABLE"] = "1"
            candidate_edges, _graph_stats = _apply_external_ranked_consensus(
                candidate_nodes, candidate_edges, stem
            )
        finally:
            globals()["TEST_DIR"] = _real_test_dir
            if _real_graph_flag is None:
                os.environ.pop("BIOHUB_FRESH_GRAPH_ENABLE", None)
            else:
                os.environ["BIOHUB_FRESH_GRAPH_ENABLE"] = _real_graph_flag

        for arm, arm_nodes, arm_edges in (
            ("control", control_nodes, control_edges),
            ("fresh_graph", candidate_nodes, candidate_edges),
        ):
            plain_edges = [
                (int(edge["source_id"]), int(edge["target_id"])) for edge in arm_edges
            ]
            row = score_sample(
                nodes_by_id_to_plain(arm_nodes), plain_edges,
                gt_nodes_plain, gt_edges_plain, t_true,
            )
            row["stem"] = stem
            row["arm"] = arm
            row["fresh_graph_edges_added"] = (
                int(_graph_stats["added_edges"]) if arm == "fresh_graph" else 0
            )
            row["t_true_source"] = (
                "estimated_number_of_nodes" if t_true is not None else "MISSING"
            )
            validator_rows_by_arm[arm].append(row)
            validator_sample_rows.append(row)

    for arm, rows_this_config in validator_rows_by_arm.items():
        if rows_this_config:
            summary = aggregate_official(rows_this_config)
            summary["arm"] = arm
            summary["n_samples"] = len(rows_this_config)
            validator_summary_rows.append(summary)
'''


EVIDENCE = r'''# Fail-closed paired complete-movie promotion evidence.
_gcd_validator = pd.read_csv(VALIDATOR_STATS_PATH)
_gcd_by_arm = {
    arm: _gcd_validator[_gcd_validator["arm"] == arm].copy()
    for arm in ("control", "fresh_graph")
}
if any(len(rows) != len(val_stems) for rows in _gcd_by_arm.values()):
    raise RuntimeError("Fresh graph candidate/control coverage is incomplete")

def _gcd_aggregate_frame(rows):
    weight = float(rows["weight"].sum())
    if weight <= 0:
        raise RuntimeError("Validator arm emitted zero edge weight")
    adjusted = float((rows["adjusted_edge_jaccard"] * rows["weight"]).sum() / weight)
    div_tp = int(rows["div_tp"].sum())
    div_fp = int(rows["div_fp"].sum())
    div_fn = int(rows["div_fn"].sum())
    denominator = div_tp + div_fp + div_fn
    division = div_tp / denominator if denominator else 0.0
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + VALIDATOR_DIVISION_WEIGHT * division,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
    }

_gcd_control = _gcd_aggregate_frame(_gcd_by_arm["control"])
_gcd_candidate = _gcd_aggregate_frame(_gcd_by_arm["fresh_graph"])
_gcd_control_movie = _gcd_by_arm["control"].set_index("stem")
_gcd_candidate_movie = _gcd_by_arm["fresh_graph"].set_index("stem")
if set(_gcd_control_movie.index) != set(_gcd_candidate_movie.index):
    raise RuntimeError("Fresh graph candidate/control movie sets differ")
_gcd_per_movie_delta = {
    stem: float(
        _gcd_candidate_movie.loc[stem, "adjusted_edge_jaccard"]
        - _gcd_control_movie.loc[stem, "adjusted_edge_jaccard"]
    )
    for stem in sorted(_gcd_control_movie.index)
}
_gcd_production_stats = pd.read_csv(RUN_STATS_PATH)
_gcd_edges_added = int(
    _gcd_production_stats["ranked_consensus_added_edges"].sum()
)
_gcd_integrity_columns = (
    "ranked_consensus_reassignment_performed",
    "ranked_consensus_node_or_coordinate_changes",
)
_gcd_integrity_passed = all(
    int(_gcd_production_stats[name].sum()) == 0
    for name in _gcd_integrity_columns
)
_gcd_eligible = bool(
    _gcd_candidate["proxy_score"] > _gcd_control["proxy_score"]
    and min(_gcd_per_movie_delta.values()) >= 0.0
    and _gcd_candidate["division_jaccard"] > _gcd_control["division_jaccard"]
    and _gcd_edges_added > 0
    and _gcd_integrity_passed
)
_gcd_submission = Path("/kaggle/working/submission.csv")
_gcd_evidence = {
    "schema_version": 1,
    "status": (
        "eligible_for_submission"
        if _gcd_eligible else "rejected_at_complete_movie_gate"
    ),
    "run_id": "948tta2-fresh-graph-v3",
    "target_public_score": 0.945,
    "source_notebook": "redoctopusk/biohub-948tta2",
    "source_notebook_sha256": "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189",
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "exact_public_replica": False,
    "metric_hack_used": False,
    "leaderboard_used_for_candidate_selection": False,
    "fresh_graph_manifest_sha256": _GCD_MANIFEST_SHA256,
    "fresh_graph_policy_sha256": _gcd_sha256(
        _GCD_ROOT / "graph-context-fresh-policy.json"
    ),
    "graph_member_count": len(_GCD_DEEP_MEMBERS),
    "graph_parameter_count_per_member": 74_732_308,
    "graph_model_sha256": [row["model_sha256"] for row in _GCD_DEEP_MEMBERS],
    "morphology_model_sha256": _GCD_POLICY["morphology_model_sha256"],
    "production_edges_added": _gcd_edges_added,
    "integrity_passed": _gcd_integrity_passed,
    "control": _gcd_control,
    "candidate": _gcd_candidate,
    "proxy_score_delta": _gcd_candidate["proxy_score"] - _gcd_control["proxy_score"],
    "minimum_movie_adjusted_edge_delta": min(_gcd_per_movie_delta.values()),
    "per_movie_adjusted_edge_delta": _gcd_per_movie_delta,
    "complete_movie_count": len(_gcd_per_movie_delta),
    "competition_submission_performed": False,
    "authorized_for_submission": _gcd_eligible,
    "submission_sha256": _gcd_sha256(_gcd_submission),
}
Path("/kaggle/working/candidate_evidence.json").write_text(
    _gcd_json.dumps(_gcd_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_gcd_json.dumps(_gcd_evidence, indent=2, sort_keys=True))
_LC_FINISHED = True
_LC_TIMER.cancel()
_lc_write_terminal("completed")
'''


def freeze_validation_selection(source: str) -> str:
    start_marker = "    # REVIEW: division-aware sample selection."
    end_marker = "    print(val_stems)\n"
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError("948TTA2 validator selection start changed")
    end_start = source.find(end_marker, start)
    if end_start < 0:
        raise RuntimeError("948TTA2 validator selection end changed")
    end = end_start + len(end_marker)
    frozen = f'''    # Predeclared before this candidate's labels or predictions were opened.
    val_stems = {list(FROZEN_VALIDATION_STEMS)!r}
    missing_frozen = sorted(set(val_stems) - set(candidates))
    if missing_frozen:
        raise RuntimeError(f"Frozen validation movies are unavailable: {{missing_frozen}}")
    print(f"VALIDATOR: using {{len(val_stems)}} predeclared complete movies")
    print(val_stems)
'''
    return source[:start] + frozen + source[end:]


def build_notebook(graph_root: Path) -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Audited 0.948-TTA2 notebook source changed")
    if sha256_file(SOURCE_METADATA) != SOURCE_METADATA_SHA256:
        raise RuntimeError("Audited 0.948-TTA2 metadata changed")
    deployment = DEPLOY["verify_dataset"](graph_root)
    if deployment.get("authorized_for_full_candidate_evaluation") is not True:
        raise RuntimeError("Fresh graph runtime is not authorized for evaluation")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    config_index = find_cell(notebook, 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"')
    config = "".join(notebook["cells"][config_index]["source"])
    config = replace_exact(
        config,
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"',
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"\n'
        'os.environ["BIOHUB_FRESH_GRAPH_ENABLE"] = "1"',
    )
    notebook["cells"][config_index]["source"] = config.splitlines(keepends=True)

    selection_index = find_cell(notebook, "VALIDATOR: held-out sample selection")
    selection = "".join(notebook["cells"][selection_index]["source"])
    notebook["cells"][selection_index]["source"] = freeze_validation_selection(
        selection
    ).splitlines(keepends=True)

    post_index = find_cell(notebook, "def filter_output_graph(")
    post = "".join(notebook["cells"][post_index]["source"])
    post = replace_exact(
        post,
        "def filter_output_graph(\n",
        ranked_helpers() + "def filter_output_graph(\n",
    )
    apply_anchor = (
        "    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n"
    )
    post = replace_exact(
        post,
        apply_anchor,
        apply_anchor + GRAPH["COMMON"]["RANKED_APPLY"],
    )
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)

    validator_index = find_cell(
        notebook, "VALIDATOR: scoring against the official metric"
    )
    validator = "".join(notebook["cells"][validator_index]["source"])
    validator = replace_exact(
        validator, BASE["VALIDATOR_LOOP_OLD"], VALIDATOR_LOOP_NEW
    )
    notebook["cells"][validator_index]["source"] = validator.splitlines(
        keepends=True
    )

    watchdog = BASE["WATCHDOG"].replace("948tta2-lsm-consensus-v1", RUN_ID)
    config_index = find_cell(
        notebook, 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"'
    )
    notebook["cells"][config_index:config_index] = [code_cell(watchdog)]

    inference_index = find_cell(notebook, "Prediction completed in")
    notebook["cells"][inference_index + 1 : inference_index + 1] = [
        markdown_cell(ATTRIBUTION),
        code_cell(model_setup(deployment["manifest_sha256"])),
    ]

    post_index = find_cell(notebook, "def filter_output_graph(")
    notebook["cells"][post_index + 1 : post_index + 1] = [code_cell(OFFLOAD_MODELS)]

    prediction_index = find_cell(notebook, "predict_val_cmd = [")
    notebook["cells"][prediction_index + 1 : prediction_index + 1] = [
        code_cell(RESTORE_MODELS)
    ]

    validator_index = find_cell(
        notebook, "VALIDATOR: scoring against the official metric"
    )
    notebook["cells"][validator_index + 1 : validator_index + 1] = [
        code_cell(EVIDENCE)
    ]

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["codex"] = {
        "run_id": RUN_ID,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "fresh_graph_manifest_sha256": deployment["manifest_sha256"],
        "frozen_validation_stems": list(FROZEN_VALIDATION_STEMS),
        "submission_command_included": False,
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    graph_root = args.graph_root.resolve()

    if TARGET_DIR.exists():
        if not args.replace:
            raise FileExistsError(f"Target already exists: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook(graph_root)
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    dataset_sources = list(base_metadata.get("dataset_sources", []))
    if RUNTIME_REF not in dataset_sources:
        dataset_sources.append(RUNTIME_REF)
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub 948TTA2 Fresh Graph v3",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "fresh-graph", "non-replica"],
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
