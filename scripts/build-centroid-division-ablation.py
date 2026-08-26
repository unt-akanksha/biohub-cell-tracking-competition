from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle" / "biohub-clean-0927-repro-v1"
TARGET_DIR = ROOT / "kaggle" / "biohub-centroid-division-ablation-v1"
NOTEBOOK_NAME = "biohub-0-927-lb.ipynb"
RUN_ID = "centroid-division-ablation-v1"


ABLATION_MARKDOWN = """## 10. Held-out centroid and safe-division ablation

This cell runs a complete-movie, label-clean 2x2 post-processing ablation on the
four validator movies. It does not change the generated test submission. The
baseline above omitted centroid refinement even though the test submission path
uses it, so the aligned arm below closes that validation gap.
"""


ABLATION_CODE = r'''# Complete-movie post-processing ablation; test output remains untouched.
import copy as _ablation_copy

ABLATION_RESULTS_PATH = WORKING_DIR / "validator_ablation_results.csv"
ABLATION_SUMMARY_PATH = WORKING_DIR / "validator_ablation_summary.json"


def _load_ablation_input(stem):
    gt_path = TRAIN_DIR / f"{stem}.geff"
    pred_path = val_pred_paths[stem]
    gt_graph = graph_from_geff(gt_path)
    gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
    t_true = read_estimated_true_node_count(gt_path)

    pred_graph = graph_from_geff(pred_path)
    raw_nodes = {}
    for item in pred_graph.node_attrs().iter_rows(named=True):
        node_id = int(item["node_id"])
        raw_nodes[node_id] = {
            "node_id": node_id,
            "t": int(item["t"]),
            "z": float(item["z"]),
            "y": float(item["y"]),
            "x": float(item["x"]),
        }
    raw_edges = []
    for item in pred_graph.edge_attrs().iter_rows(named=True):
        edge_prob = item.get("edge_prob") if hasattr(item, "get") else None
        raw_edges.append({
            "source_id": int(item["source_id"]),
            "target_id": int(item["target_id"]),
            "edge_prob": None if edge_prob is None else float(edge_prob),
        })
    return {
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
        "gt_nodes": gt_nodes_plain,
        "gt_edges": gt_edges_plain,
        "t_true": t_true,
    }


def _run_ablation_arm(arm, refine_centers, safe_divisions, inputs):
    arm_rows = []
    stage_totals = {}
    saved_test_dir = TEST_DIR
    saved_safe_divisions = OUTPUT_SAFE_DIVISIONS
    globals()["TEST_DIR"] = TRAIN_DIR
    globals()["OUTPUT_SAFE_DIVISIONS"] = bool(safe_divisions)
    try:
        for stem in val_stems:
            item = inputs[stem]
            nodes = _ablation_copy.deepcopy(item["raw_nodes"])
            edges = _ablation_copy.deepcopy(item["raw_edges"])
            if refine_centers:
                nodes = refine_all_centroids(nodes, stem)
            processed_nodes, processed_edges, stage_stats = filter_output_graph(
                nodes,
                edges,
                dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            row = score_sample(
                nodes_by_id_to_plain(processed_nodes),
                [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges],
                item["gt_nodes"],
                item["gt_edges"],
                item["t_true"],
            )
            row.update({
                "arm": arm,
                "stem": stem,
                "centroid_refinement": bool(refine_centers),
                "output_safe_divisions": bool(safe_divisions),
            })
            for key in (
                "safe_division_geometric_candidates",
                "safe_division_candidates",
                "safe_divisions_added",
                "deepcenter_safe_div_rejected",
            ):
                value = int(stage_stats.get(key, 0))
                row[key] = value
                stage_totals[key] = stage_totals.get(key, 0) + value
            arm_rows.append(row)
    finally:
        globals()["TEST_DIR"] = saved_test_dir
        globals()["OUTPUT_SAFE_DIVISIONS"] = saved_safe_divisions

    summary = aggregate_official(arm_rows)
    summary.update({
        "arm": arm,
        "n_samples": len(arm_rows),
        "centroid_refinement": bool(refine_centers),
        "output_safe_divisions": bool(safe_divisions),
        **stage_totals,
    })
    return arm_rows, summary


if VALIDATOR_ENABLE and val_stems and validator_sample_rows:
    missing_ablation = [stem for stem in val_stems if stem not in val_pred_paths]
    if missing_ablation:
        raise RuntimeError(f"Ablation inputs are missing validator predictions: {missing_ablation}")

    ablation_inputs = {stem: _load_ablation_input(stem) for stem in val_stems}
    baseline_rows = []
    for existing in validator_sample_rows:
        row = dict(existing)
        row.update({
            "arm": "centroid_off__safe_div_on",
            "centroid_refinement": False,
            "output_safe_divisions": True,
        })
        baseline_rows.append(row)
    baseline_summary = aggregate_official(baseline_rows)
    baseline_summary.update({
        "arm": "centroid_off__safe_div_on",
        "n_samples": len(baseline_rows),
        "centroid_refinement": False,
        "output_safe_divisions": True,
    })

    all_ablation_rows = list(baseline_rows)
    ablation_summaries = [baseline_summary]
    for arm, refine_centers, safe_divisions in (
        ("centroid_off__safe_div_off", False, False),
        ("centroid_on__safe_div_on", True, True),
        ("centroid_on__safe_div_off", True, False),
    ):
        rows, arm_summary = _run_ablation_arm(
            arm, refine_centers, safe_divisions, ablation_inputs
        )
        all_ablation_rows.extend(rows)
        ablation_summaries.append(arm_summary)

    baseline_proxy = float(baseline_summary["proxy_score"])
    for item in ablation_summaries:
        item["delta_proxy_vs_unaligned_baseline"] = float(item["proxy_score"]) - baseline_proxy
    best_summary = max(ablation_summaries, key=lambda item: float(item["proxy_score"]))

    with ABLATION_RESULTS_PATH.open("w", newline="") as handle:
        fieldnames = sorted({key for row in all_ablation_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_ablation_rows)

    ablation_payload = {
        "design": "complete-movie 2x2 centroid-refinement x safe-division ablation",
        "heldout_stems": list(val_stems),
        "leakage_guard": "train stems overlapping competition test stems excluded before selection",
        "baseline_note": "original validator path omitted centroid refinement",
        "arms": ablation_summaries,
        "best_arm": best_summary["arm"],
        "best_proxy_score": best_summary["proxy_score"],
    }
    ABLATION_SUMMARY_PATH.write_text(
        json.dumps(ablation_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 78)
    print("HELD-OUT POST-PROCESSING ABLATION")
    print("=" * 78)
    for item in ablation_summaries:
        print(
            f"{item['arm']:<31} adjusted={item['adjusted_edge_jaccard']:.6f} "
            f"division={item['division_jaccard']:.6f} proxy={item['proxy_score']:.6f} "
            f"delta={item['delta_proxy_vs_unaligned_baseline']:+.6f}"
        )
    print(f"Best arm: {best_summary['arm']}")
    print(f"Wrote {ABLATION_RESULTS_PATH}")
    print(f"Wrote {ABLATION_SUMMARY_PATH}")
else:
    raise RuntimeError("Ablation requires the enabled complete-movie validator outputs.")
'''


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)

    source_notebook = json.loads(
        (SOURCE_DIR / NOTEBOOK_NAME).read_text(encoding="ascii")
    )
    watchdog = "".join(source_notebook["cells"][0]["source"])
    old_run_id = "_BIOHUB_RUN_ID = 'public-0927-clean-repro-v2'"
    new_run_id = f"_BIOHUB_RUN_ID = '{RUN_ID}'"
    if old_run_id not in watchdog:
        raise RuntimeError("Expected source watchdog run ID was not found")
    source_notebook["cells"][0]["source"] = watchdog.replace(
        old_run_id, new_run_id, 1
    ).splitlines(keepends=True)

    manifest_index = next(
        i
        for i, cell in enumerate(source_notebook["cells"])
        if "PIPELINE MANIFEST -- resolved state" in "".join(cell.get("source", []))
    )
    source_notebook["cells"][manifest_index:manifest_index] = [
        _markdown_cell(ABLATION_MARKDOWN),
        _code_cell(ABLATION_CODE),
    ]
    for cell in source_notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    target_notebook = TARGET_DIR / NOTEBOOK_NAME
    target_notebook.write_text(
        json.dumps(source_notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="ascii",
    )

    metadata = json.loads(
        (SOURCE_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata.update({
        "id": "indarkarhana/biohub-centroid-division-ablation-v1",
        "title": "Biohub Centroid Division Ablation v1",
        "code_file": NOTEBOOK_NAME,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
    })
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="ascii"
    )
    print(target_notebook)


if __name__ == "__main__":
    main()
