"""Offline post-process configuration screen, and the test that decides whether
it may be trusted at all.

The published pipeline's `filter_output_graph` reaches the movie images twice:
`deepcenter_heatmap_for_frame` (the veto that gates gap closures and safe
divisions) and `refine_synthetic_midpoint` (synthetic gap refinement). Neither
can run here, because the images are the ~87 GB this project deliberately never
downloaded. Offline, both must be disabled.

That makes this a DIFFERENT pipeline from the kernel's, not a reproduction. Its
absolute scores are therefore meaningless as an estimate of anything. The only
question worth asking is narrower: does it rank configurations in the same order
the kernel does? If it does, it is a free, unlimited pre-screen that can shortlist
candidates for a faithful but expensive in-kernel confirmation. If it does not,
it must be discarded rather than reasoned around.

`fidelity_check()` answers that question against evidence that already exists:
the 2026-09-10 complete-movie run scored the four diagnostic movies at a pooled
0.9448387313 under the patched official scorer with the veto active. Running the
same base configuration here, with the veto off, must be compared against that
number before any screening result is quoted.

Nothing here selects a configuration or touches a submission.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POSTPROCESS = ROOT / ".biohub/cache/public-d4-correction-v1/public-postprocess-d4-corrected.py"
RESOLVED_CONFIG = ROOT / ".biohub/cache/public-d4-full-movie-v1-bundle/resolved-public-config.json"
TRUTH_ROOT = ROOT / ".biohub/cache/competition-train-geffs-packed-v1/train"
SEPT10_OUTPUT = ROOT / ".biohub/cache/public-d4-full-movie-v1-output"

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
MAX_DISTANCE_UM = 7.0

# The 2026-09-10 pooled official score of the ORIGINAL arm on the four complete
# diagnostic movies, produced with the DeepCenter veto active.
SEPT10_ORIGINAL_POOLED = 0.9448387313

# Image-dependent stages that cannot run without the movie volumes.
OFFLINE_DISABLED = {
    "USE_DEEPCENTER_VETO": False,
    "REQUIRE_DEEPCENTER_VETO": False,
    "DEEPCENTER_GAP_VETO": False,
    "DEEPCENTER_SAFE_DIV_VETO": False,
    "GAP_REFINE_SYNTHETIC": False,
}


def _named_definitions(source: str, names) -> str:
    from research.public_d4_preflight import named_definitions

    return named_definitions(source, tuple(names))


def build_postprocess(overrides: dict | None = None):
    """Return (filter_output_graph, resolved_config) with image stages disabled."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    config = dict(json.loads(RESOLVED_CONFIG.read_text())["globals"])
    config.update(OFFLINE_DISABLED)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(overrides)

    env = dict(
        config,
        np=np,
        math=math,
        json=json,
        os=__import__("os"),
        linear_sum_assignment=linear_sum_assignment,
        cKDTree=cKDTree,
        zarr=None,
        blosc2=None,
        TEST_DIR=Path("/offline/no-images"),
        VOXEL_SCALE_UM=VOXEL_SCALE_UM,
    )
    source = POSTPROCESS.read_text()
    needed = (
        "edge_distance_um", "_position_um", "point_distance_um", "node_point",
        "edge_sort_key", "_next_node_id", "_single_predecessor_map",
        "_single_successor_map", "_dc_cache_trim", "_dc_normalize_dynamic_range",
        "_dc_pool_frame_xy", "read_test_frame", "deepcenter_heatmap_for_frame",
        "deepcenter_score_point", "deepcenter_accept_repair_point",
        "refine_synthetic_midpoint", "motion_relink_edges", "close_single_frame_gaps",
        "recover_strict_gap2", "add_safe_divisions_postlink",
        "filter_short_track_components", "linefit_smooth_output_graph",
        "filter_output_graph",
    )
    exec(compile(_named_definitions(source, needed), "offline-postprocess", "exec"), env)
    return env["filter_output_graph"], config


def load_scorer():
    import hashlib

    folder = ROOT / ".biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot"
    expected = {
        "metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
        "division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
    }
    for name, digest in expected.items():
        payload = (folder / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Patched official scorer changed")
    name = "_screen_official_scorer"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    return importlib.import_module(name + ".metrics")


def prediction_graph(nodes_by_id, edges):
    import polars as pl
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    ids = sorted(nodes_by_id, key=int)
    mapped = graph.bulk_add_nodes(
        [{k: float(nodes_by_id[i][k]) if k != "t" else int(nodes_by_id[i][k])
          for k in ("t", "z", "y", "x")} for i in ids]
    )
    mapping = dict(zip((int(i) for i in ids), mapped))
    graph.bulk_add_edges(
        [dict(source_id=mapping[int(e["source_id"])], target_id=mapping[int(e["target_id"])])
         for e in edges]
    )
    return graph


def score_movie(nodes_by_id, edges, stem, scorer=None):
    import tracksdata as td
    from geff import GeffMetadata

    scorer = scorer or load_scorer()
    truth_path = TRUTH_ROOT / f"{stem}.geff"
    truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
    graph = prediction_graph(nodes_by_id, edges)
    result = scorer.evaluate(graph, truth, scale=VOXEL_SCALE_UM, max_distance=MAX_DISTANCE_UM)
    count = float(GeffMetadata.read(str(truth_path)).extra["estimated_number_of_nodes"])
    row = dict(scorer.per_sample_metrics(result, count, scorer.node_recall(graph, truth)))
    row["stem"] = stem
    row["embryo"] = stem.split("_")[0]
    return row


def run_config(raw_graphs: dict, overrides: dict | None, label: str, scorer=None):
    """Post-process every raw graph under one configuration and score it."""
    import copy as _copy

    filter_output_graph, _ = build_postprocess(overrides)
    scorer = scorer or load_scorer()
    rows = []
    for stem, (nodes_by_id, edges) in raw_graphs.items():
        nodes, processed_edges, _stats = filter_output_graph(
            _copy.deepcopy(nodes_by_id), _copy.deepcopy(edges),
            dataset=stem, deepcenter_bundle=None,
        )
        row = score_movie(nodes, processed_edges, stem, scorer=scorer)
        row["config"] = label
        rows.append(row)
    return rows


def load_sept10_raw_graphs(arm: str = "original") -> dict:
    """The four 2026-09-10 diagnostic movies, as raw pre-postprocess graphs."""
    graphs = {}
    from research.public_d4_full_movie import STEMS

    for stem in STEMS:
        payload = json.loads((SEPT10_OUTPUT / f"{stem}-{arm}" / "pre-postprocess.json").read_text())
        nodes = {int(i): dict(n) for i, n in payload["nodes"].items()}
        graphs[stem] = (nodes, list(payload["edges"]))
    return graphs


def fidelity_check() -> dict:
    """Quantify what disabling the image stages costs, against known evidence.

    Runs the unmodified base configuration offline on the four 2026-09-10 movies
    and compares the pooled official score with the 0.9448387313 that the same
    configuration produced in-pipeline with the veto active.
    """
    scorer = load_scorer()
    graphs = load_sept10_raw_graphs("original")
    rows = run_config(graphs, None, "offline_base", scorer=scorer)
    summary = scorer.summarise(rows)
    delta = summary["score"] - SEPT10_ORIGINAL_POOLED
    return {
        "offline_pooled": summary["score"],
        "in_pipeline_pooled": SEPT10_ORIGINAL_POOLED,
        "delta": delta,
        "abs_delta": abs(delta),
        "per_movie": {r["stem"]: scorer.summarise([r])["score"] for r in rows},
        "per_movie_rows": rows,
        "image_stages_disabled": sorted(OFFLINE_DISABLED),
        "screen_is_a_reproduction": False,
        "note": (
            "A small delta means the disabled image stages barely move the base "
            "configuration, which is weak evidence the screen may rank nearby "
            "configurations correctly. It is not proof; ranking agreement against "
            "a known in-kernel sweep is the real test."
        ),
    }
