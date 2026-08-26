#!/usr/bin/env python
"""Clean complete-movie gate for LSM-FM nodes plus official association."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from association_bridge import DetectorCandidate, predict_movie_detections
    from hybrid_linker import (
        AssociationConsensusConfig,
        PRIMARY_CHECKPOINT_SHA256,
        SECONDARY_CHECKPOINT_SHA256,
        predict_video_consensus,
        sha256_file,
        validate_checkpoint,
    )
except ModuleNotFoundError:
    from research.lsm_fm_detection.association_bridge import (
        DetectorCandidate,
        predict_movie_detections,
    )
    from research.lsm_fm_detection.hybrid_linker import (
        AssociationConsensusConfig,
        PRIMARY_CHECKPOINT_SHA256,
        SECONDARY_CHECKPOINT_SHA256,
        predict_video_consensus,
        sha256_file,
        validate_checkpoint,
    )


ACCEPTANCE_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
PUBLIC_COMPARATOR_SCORE = 0.9294432421394134
PUBLIC_PREFIX_SCORE = {
    "44b6": 0.8910900942367352,
    "6bba": 0.9615550101410081,
}
PREFIX_REGRESSION_MIN = -0.01

CANDIDATES = {
    "feature24_control": DetectorCandidate("feature24_control", 1.0, 0.0),
    "feature36_control": DetectorCandidate("feature36_control", 0.0, 1.0),
    "feature36_log_quadratic": DetectorCandidate(
        "feature36_log_quadratic", 0.0, 1.0, "log_quadratic"
    ),
    "equal_ensemble_control": DetectorCandidate(
        "equal_ensemble_control", 0.5, 0.5
    ),
    "equal_ensemble_log_quadratic": DetectorCandidate(
        "equal_ensemble_log_quadratic", 0.5, 0.5, "log_quadratic"
    ),
}


def selected_detector(path: Path) -> tuple[DetectorCandidate, dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("detector ensemble result is not complete")
    if result.get("selection_passed") is not True:
        raise ValueError("detector ensemble did not pass its selection gate")
    if result.get("acceptance_opened") is not True:
        raise ValueError("detector ensemble acceptance remained sealed")
    if result.get("promotion_passed") is not True:
        raise ValueError("detector ensemble did not pass clean node acceptance")
    if result.get("public_leaderboard_used_for_selection") is not False:
        raise ValueError("detector selection provenance is not clean")
    if result.get("public_predictions_copied") is not False:
        raise ValueError("detector result copied public predictions")
    if result.get("competition_submission_performed") is not False:
        raise ValueError("detector experiment unexpectedly submitted")
    name = str(result.get("selected_candidate"))
    if name not in CANDIDATES:
        raise ValueError(f"unknown selected detector candidate: {name}")
    return CANDIDATES[name], result


def validated_training_result(model_path: Path, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("detector training result is not complete")
    if result.get("validation_overlap") != []:
        raise ValueError("detector checkpoint has validation overlap")
    if result.get("public_predictions_copied") is not False:
        raise ValueError("detector checkpoint provenance is not independent")
    if sha256_file(model_path) != result.get("best_weight_sha256"):
        raise ValueError("learned detector checkpoint hash mismatch")
    return result


def load_official_modules(source_root: Path) -> tuple[Any, Any, Any]:
    source_root = source_root.resolve()
    scripts = source_root / "scripts"
    package = source_root / "src"
    required = (
        source_root / "LICENSE",
        scripts / "predict_unet_transformer.py",
        scripts / "evaluate.py",
        package / "tracking_cellmot" / "metrics.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official source is incomplete: {missing}")
    for path in (str(package), str(scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)
    predict = importlib.import_module("predict_unet_transformer")
    evaluator = importlib.import_module("evaluate")
    metrics = importlib.import_module("tracking_cellmot.metrics")
    return predict, evaluator, metrics


def canonical_graph_sha256(graph: Any) -> str:
    nodes = sorted(
        (
            int(row["node_id"]),
            int(row["t"]),
            round(float(row["z"]), 6),
            round(float(row["y"]), 6),
            round(float(row["x"]), 6),
        )
        for row in graph.node_attrs().iter_rows(named=True)
    )
    edges = sorted(
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    payload = json.dumps(
        {"nodes": nodes, "edges": edges},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def graph_from_geff(path: Path) -> Any:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def solve_ilp(official_predict: Any, coords: np.ndarray, edges: list[tuple]) -> Any:
    graph = official_predict.build_graph(coords, edges)
    if graph.num_edges() == 0:
        return graph
    solver = official_predict.td.solvers.ILPSolver(
        edge_weight=-1.0 * official_predict.td.EdgeAttr("edge_prob"),
        appearance_weight=0.0,
        disappearance_weight=1.5,
        division_weight=1.0,
    )
    with official_predict.suppress_output():
        return solver.solve(graph)


def prefix_summaries(rows: Sequence[dict[str, Any]], metrics: Any) -> dict[str, Any]:
    return {
        prefix: metrics.summarise(
            [row for row in rows if str(row["name"]).startswith(prefix + "_")]
        )
        for prefix in PUBLIC_PREFIX_SCORE
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--primary-weights", type=Path, required=True)
    parser.add_argument("--secondary-weights", type=Path, required=True)
    parser.add_argument("--feature24-base", type=Path, required=True)
    parser.add_argument("--feature24-base-sha256", required=True)
    parser.add_argument("--feature24-model", type=Path, required=True)
    parser.add_argument("--feature24-training-result", type=Path, required=True)
    parser.add_argument("--feature36-base", type=Path, required=True)
    parser.add_argument("--feature36-base-sha256", required=True)
    parser.add_argument("--feature36-model", type=Path, required=True)
    parser.add_argument("--feature36-training-result", type=Path, required=True)
    parser.add_argument("--ensemble-result", type=Path, required=True)
    parser.add_argument("--public-comparator-geffs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("hybrid association validation requires CUDA")
    candidate, detector_selection = selected_detector(args.ensemble_result)
    feature24_training = validated_training_result(
        args.feature24_model, args.feature24_training_result
    )
    feature36_training = validated_training_result(
        args.feature36_model, args.feature36_training_result
    )
    validate_checkpoint(args.primary_weights, PRIMARY_CHECKPOINT_SHA256)
    validate_checkpoint(args.secondary_weights, SECONDARY_CHECKPOINT_SHA256)
    official_predict, official_evaluator, official_metrics = load_official_modules(
        args.official_source_root
    )
    device = torch.device("cuda")
    primary, primary_window, primary_downsample = official_predict.load_model(
        args.primary_weights, device
    )
    secondary, secondary_window, secondary_downsample = official_predict.load_model(
        args.secondary_weights, device
    )
    association_config = AssociationConsensusConfig()
    if (
        primary_window != association_config.window_size
        or secondary_window != association_config.window_size
        or tuple(primary_downsample) != association_config.downsample
        or tuple(secondary_downsample) != association_config.downsample
    ):
        raise ValueError("association checkpoints have incompatible grids")

    try:
        from lsm_fm_image_only_model import build_lsm_fm_detector as build_feature24
        from lsm_fm_image_text_model import build_lsm_fm_detector as build_feature36
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import (
            build_lsm_fm_detector as build_feature36,
        )
        from research.lsm_fm_detection.model import (
            build_lsm_fm_detector as build_feature24,
        )

    detectors = {
        "feature24": build_feature24(
            args.feature24_base, expected_sha256=args.feature24_base_sha256
        ),
        "feature36": build_feature36(
            args.feature36_base, expected_sha256=args.feature36_base_sha256
        ),
    }
    for name, model_path in (
        ("feature24", args.feature24_model),
        ("feature36", args.feature36_model),
    ):
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        detectors[name].load_state_dict(payload["state_dict"], strict=True)
        detectors[name].requires_grad_(False).eval().to(device)
    linkers = {"primary": primary, "secondary": secondary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    comparator_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}

    for stem in ACCEPTANCE_STEMS:
        sample_path = args.competition_dir / "train" / f"{stem}.zarr"
        cache = predict_movie_detections(
            detectors,
            sample_path,
            candidate=candidate,
            device=device,
            batch_size=args.batch_size,
        )
        coords, edges, manifest = predict_video_consensus(
            official_predict,
            linkers,
            sample_path,
            device,
            cache,
            config=association_config,
        )
        graph = solve_ilp(official_predict, coords, edges)
        output_path = predictions / f"{stem}.geff"
        official_predict.save_graph(graph, output_path)
        candidate_hashes[stem] = canonical_graph_sha256(graph)
        comparator = graph_from_geff(args.public_comparator_geffs / f"{stem}.geff")
        comparator_hashes[stem] = canonical_graph_sha256(comparator)
        manifests[stem] = {
            **manifest,
            "ilp_nodes": graph.num_nodes(),
            "ilp_edges": graph.num_edges(),
            "candidate_graph_sha256": candidate_hashes[stem],
            "comparator_graph_sha256": comparator_hashes[stem],
        }
        partial = {
            "selected_detector": candidate.name,
            "completed_stems": list(manifests),
            "manifests": manifests,
        }
        (args.output_dir / "hybrid_association_partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("hybrid association validation reached its wall guard")

    rows, skipped = official_evaluator.evaluate_pairs(
        predictions, args.competition_dir / "train", max_distance=7.0
    )
    if skipped or len(rows) != len(ACCEPTANCE_STEMS):
        raise RuntimeError(f"complete-movie scoring failed: skipped={skipped}")
    for row, stem in zip(rows, sorted(ACCEPTANCE_STEMS), strict=True):
        row["name"] = stem
    summary = official_metrics.summarise(rows)
    by_prefix = prefix_summaries(rows, official_metrics)
    prefix_deltas = {
        prefix: float(by_prefix[prefix]["score"]) - baseline
        for prefix, baseline in PUBLIC_PREFIX_SCORE.items()
    }
    graph_differs = any(
        candidate_hashes[stem] != comparator_hashes[stem]
        for stem in ACCEPTANCE_STEMS
    )
    promotion_passed = (
        float(summary["score"]) > PUBLIC_COMPARATOR_SCORE
        and min(prefix_deltas.values()) >= PREFIX_REGRESSION_MIN
        and graph_differs
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "selected_detector": candidate.name,
        "detector_selection_result_sha256": sha256_file(args.ensemble_result),
        "detector_selection_promotion_passed": detector_selection["promotion_passed"],
        "feature24_training_result_sha256": sha256_file(
            args.feature24_training_result
        ),
        "feature36_training_result_sha256": sha256_file(
            args.feature36_training_result
        ),
        "feature24_training_status": feature24_training["status"],
        "feature36_training_status": feature36_training["status"],
        "association_config": association_config.__dict__,
        "summary": summary,
        "by_prefix": by_prefix,
        "prefix_deltas_vs_public_comparator": prefix_deltas,
        "public_comparator_score": PUBLIC_COMPARATOR_SCORE,
        "graph_differs_from_public_comparator": graph_differs,
        "promotion_passed": promotion_passed,
        "manifests": manifests,
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
    }
    output = args.output_dir / "lsm_fm_hybrid_association_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LSM-FM HYBRID ASSOCIATION COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
