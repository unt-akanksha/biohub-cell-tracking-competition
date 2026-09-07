#!/usr/bin/env python3
"""Build the precommitted V28 detector plus graph-context ensemble candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
PEAK = runpy.run_path(str(ROOT / "scripts/build-peak-rank-submission-candidate.py"))
GRAPH = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-consensus-submission-candidate.py")
)
GRAPH_DATA = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-consensus-division-dataset.py")
)
TARGET_ID = "biohub-peak-rank-graph-context-composition-v30"
TARGET = ROOT / "kaggle" / TARGET_ID
NOTEBOOK = TARGET / f"{TARGET_ID}.ipynb"
RUN_ID = "peak-rank-graph-context-composition-v30"
PEAK_RUNTIME_REF = (
    "indarkarhana/biohub-peak-rank-xl-hard-mined-temporal-snr-pair-"
    "validation-runtime-v28"
)
GRAPH_RUNTIME_REF = "indarkarhana/biohub-graph-context-consensus-division-v1"


ATTRIBUTION = """## Project composition: detector ensemble plus graph-context division

This candidate composes two independently gated project components without
changing either one. Node generation uses the fixed 213.6M-parameter V28
equal-logit detector pair. Association and graph construction retain the
attributed RedOctopusk/Pilkwang lineage. The additive division stage uses the
fresh-seed 74.7M graph-context models whose equal-rank membership was frozen
before its ensemble-level sealed audit, plus the independent morphology voter.

No public prediction is copied, no component is selected on the leaderboard,
and no composition weight is tuned. The graph stage may add at most one
geometry-eligible edge per movie when both rankings select the same parent.
The detector and graph components must first pass their standalone gates; this
composition must then beat both under complete-movie patched official scoring.
"""


OFFLOAD_GRAPH_MODELS = r'''# Release graph GPU memory during detector validation inference.
for _gcd_group in _GCD_MODELS_BY_DEVICE:
    for _gcd_index, _gcd_model in _gcd_group:
        _gcd_model.to("cpu")
for _gcd_device in _GCD_DEVICES:
    with _gcd_torch.cuda.device(_gcd_device):
        _gcd_torch.cuda.empty_cache()
print("Graph-context models parked on CPU for detector validation inference.")
'''


RESTORE_GRAPH_MODELS = r'''# Restore the immutable graph ensemble for held-out scoring.
for _gcd_device_index, _gcd_group in enumerate(_GCD_MODELS_BY_DEVICE):
    for _gcd_index, _gcd_model in _gcd_group:
        _gcd_model.to(_GCD_DEVICES[_gcd_device_index])
print("Graph-context models restored for complete-movie validation.")
'''


GRAPH_EVIDENCE = r'''# Bind the independently promoted graph component into candidate evidence.
_pr_graph_stats = pd.read_csv(RUN_STATS_PATH)
for _pr_graph_column in (
    "ranked_consensus_geometric_candidates",
    "ranked_consensus_geometry_eligible_candidates",
    "ranked_consensus_ranking_agreed",
    "ranked_consensus_added_edges",
):
    if _pr_graph_column not in _pr_graph_stats:
        raise RuntimeError(f"Graph-context run statistic is missing: {_pr_graph_column}")
_pr_evidence.update({
    "component_order": ["peak_rank_detector_v28", "graph_context_division_v2"],
    "independent_component_promotion_required": True,
    "graph_runtime_manifest_sha256": _GCD_MANIFEST_SHA256,
    "graph_policy_sha256": _gcd_sha256(_GCD_ROOT / "graph-context-consensus-policy.json"),
    "graph_policy": _GCD_POLICY["graph_context_policy"],
    "graph_policy_contract": _GCD_POLICY.get("policy_contract"),
    "graph_policy_unit_audited": _GCD_POLICY.get("policy_unit_audited"),
    "graph_member_count": len(_GCD_DEEP_MEMBERS),
    "graph_parameter_count_per_member": 74_732_308,
    "graph_ranked_candidates": int(
        _pr_graph_stats["ranked_consensus_geometric_candidates"].sum()
    ),
    "graph_ranked_agreements": int(
        _pr_graph_stats["ranked_consensus_ranking_agreed"].sum()
    ),
    "graph_ranked_edges_added": int(
        _pr_graph_stats["ranked_consensus_added_edges"].sum()
    ),
})
Path("/kaggle/working/candidate_evidence.json").write_text(
    _pr_json.dumps(_pr_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
'''


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def find_cell(notebook: dict, pattern: str) -> int:
    matches = [
        index
        for index, cell in enumerate(notebook["cells"])
        if pattern in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"composition source drift for {pattern!r}: {matches}")
    return matches[0]


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"composition source drift for {old!r}")
    return source.replace(old, new, 1)


def build_notebook(peak_manifest_sha256: str, graph_manifest_sha256: str) -> dict:
    peak_globals = PEAK["build_notebook"].__globals__
    peak_globals.update(
        {
            "CANDIDATE_RUN_ID": RUN_ID,
            "ATTRIBUTION": ATTRIBUTION,
        }
    )
    notebook = PEAK["build_notebook"](peak_manifest_sha256)

    inference_index = find_cell(notebook, "Prediction completed in")
    graph_setup = GRAPH["MODEL_SETUP_TEMPLATE"].replace(
        "__MANIFEST_SHA256__", graph_manifest_sha256
    )
    notebook["cells"][inference_index + 1 : inference_index + 1] = [
        markdown_cell(ATTRIBUTION),
        code_cell(graph_setup),
    ]

    post_index = find_cell(notebook, "def motion_relink_edges(")
    post = "".join(notebook["cells"][post_index]["source"])
    post = replace_once(
        post,
        "def filter_output_graph(\n",
        GRAPH["RANKED_HELPERS"] + "def filter_output_graph(\n",
    )
    post = replace_once(
        post,
        "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
        GRAPH["COMMON"]["RANKED_APPLY"]
        + "    return nodes_by_id, edges, stats\n\n\nDEEPCENTER_VETO_DETECTOR",
    )
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)
    notebook["cells"][post_index + 1 : post_index + 1] = [code_cell(OFFLOAD_GRAPH_MODELS)]

    validator_inference_index = find_cell(notebook, "predict_val_cmd = [")
    notebook["cells"][validator_inference_index + 1 : validator_inference_index + 1] = [
        code_cell(RESTORE_GRAPH_MODELS)
    ]
    evidence_index = find_cell(notebook, "_PR_FINISHED = True")
    evidence = "".join(notebook["cells"][evidence_index]["source"])
    evidence = replace_once(
        evidence,
        "_PR_FINISHED = True\n",
        GRAPH_EVIDENCE + "_PR_FINISHED = True\n",
    )
    notebook["cells"][evidence_index]["source"] = evidence.splitlines(keepends=True)
    notebook["metadata"]["codex"].update(
        {
            "candidate_run_id": RUN_ID,
            "component_order": ["peak_rank_detector_v28", "graph_context_division_v2"],
            "independent_component_promotion_required": True,
            "graph_runtime_manifest_sha256": graph_manifest_sha256,
        }
    )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-runtime-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    peak_hash = PEAK["verify_promoted_runtime"](args.peak_runtime_root)
    graph_verified = GRAPH_DATA["verify_dataset"](args.graph_root)
    if TARGET.exists():
        if not args.replace:
            raise FileExistsError(TARGET)
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    NOTEBOOK.write_text(
        json.dumps(
            build_notebook(peak_hash, graph_verified["manifest_sha256"]),
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        encoding="ascii",
    )
    source_metadata = json.loads(PEAK["SOURCE_METADATA"].read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub Peak Rank plus Graph Context Composition v30",
        "code_file": NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            *source_metadata["dataset_sources"],
            PEAK_RUNTIME_REF,
            GRAPH_RUNTIME_REF,
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
