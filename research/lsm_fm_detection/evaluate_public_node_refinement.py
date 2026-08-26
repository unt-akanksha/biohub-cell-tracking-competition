#!/usr/bin/env python
"""Refine frozen public-graph nodes with an independent LSM-FM heatmap.

The experiment preserves every public graph node, edge, and identifier.  Only
node coordinates may move, using one globally selected blend toward a local
LSM-FM probability centroid.  Selection and acceptance movies remain disjoint.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from evaluate_localization_refinement import sha256_file, summarize_rows
    from evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        FramePeaks,
        SPATIAL_DOWNSAMPLE,
        graph_from_geff,
        graph_points_by_frame,
        score_predictions,
    )
    from inference import predict_probability_batch
    from localization_refinement import refine_peaks_weighted
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.lsm_fm_detection.evaluate_localization_refinement import (
        sha256_file,
        summarize_rows,
    )
    from research.lsm_fm_detection.localization_refinement import (
        refine_peaks_weighted,
    )
    from research.spatialdino_detection.inference import predict_probability_batch
    from research.spatialdino_detection.train_pu_detector import (
        normalize_spatialdino_frame,
    )
    from research.spotiflow_biohub.evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        FramePeaks,
        SPATIAL_DOWNSAMPLE,
        graph_from_geff,
        graph_points_by_frame,
        score_predictions,
    )


EXPECTED_DETECTOR_PARAMETERS = 35_072_515
CONTROL_NAME = "public_control"
MAXIMUM_MOVIE_RECALL_REGRESSION = 0.002


@dataclass(frozen=True)
class PublicNodeStrategy:
    name: str
    radius: int
    probability_power: float
    blend: float

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("radius cannot be negative")
        if self.probability_power <= 0:
            raise ValueError("probability_power must be positive")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must lie in [0, 1]")


STRATEGIES = (
    PublicNodeStrategy(CONTROL_NAME, 0, 2.0, 0.0),
    PublicNodeStrategy("public_lsm_r1_p2_b025", 1, 2.0, 0.25),
    PublicNodeStrategy("public_lsm_r1_p2_b050", 1, 2.0, 0.50),
    PublicNodeStrategy("public_lsm_r1_p2_b075", 1, 2.0, 0.75),
    PublicNodeStrategy("public_lsm_r1_p2_b100", 1, 2.0, 1.00),
    PublicNodeStrategy("public_lsm_r2_p2_b025", 2, 2.0, 0.25),
    PublicNodeStrategy("public_lsm_r2_p2_b050", 2, 2.0, 0.50),
    PublicNodeStrategy("public_lsm_r2_p2_b075", 2, 2.0, 0.75),
    PublicNodeStrategy("public_lsm_r2_p2_b100", 2, 2.0, 1.00),
)


def refine_public_points(
    probability: np.ndarray,
    points_input: np.ndarray,
    strategy: PublicNodeStrategy,
) -> np.ndarray:
    """Blend public nodes toward a local probability centroid, in bounds."""

    heatmap = np.asarray(probability, dtype=np.float32)
    points = np.asarray(points_input, dtype=np.float32).reshape(-1, 3)
    if heatmap.ndim != 3 or not np.isfinite(heatmap).all():
        raise ValueError("probability must be a finite 3D volume")
    if not np.isfinite(points).all():
        raise ValueError("points_input must be finite")
    if not len(points) or strategy.blend == 0.0:
        return points.copy()
    target = refine_peaks_weighted(
        heatmap,
        points,
        radius=strategy.radius,
        probability_power=strategy.probability_power,
    )
    result = points + np.float32(strategy.blend) * (target - points)
    return np.clip(result, 0.0, np.asarray(heatmap.shape, dtype=np.float32) - 1.0)


def select_public_node_strategy(
    summaries: dict[str, dict[str, Any]],
    *,
    control_name: str = CONTROL_NAME,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Require a strict matched-node gain and bounded per-movie regression."""

    if control_name not in summaries:
        raise ValueError("control summary is missing")
    control = summaries[control_name]
    control_rows = {
        str(row["stem"]): float(row["candidate"]["annotated_node_recall"])
        for row in control["rows"]
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for name, summary in summaries.items():
        deltas = [
            float(row["candidate"]["annotated_node_recall"])
            - control_rows[str(row["stem"])]
            for row in summary["rows"]
        ]
        strict_match_gain = int(summary["matched_gt_nodes"]) > int(
            control["matched_gt_nodes"]
        )
        passed = (
            name != control_name
            and strict_match_gain
            and min(deltas) >= -MAXIMUM_MOVIE_RECALL_REGRESSION
        )
        diagnostics[name] = {
            "strict_matched_node_gain": strict_match_gain,
            "minimum_movie_recall_delta": min(deltas),
            "selection_passed": passed,
        }
        if passed:
            eligible.append(name)
    if not eligible:
        return None, diagnostics
    return (
        max(
            eligible,
            key=lambda name: (
                int(summaries[name]["matched_gt_nodes"]),
                float(summaries[name]["worst_movie_recall"]),
                -float(summaries[name]["mean_movie_match_distance_um"]),
                name,
            ),
        ),
        diagnostics,
    )


def predict_strategies(
    model,
    sample_path: Path,
    public_graph: Path,
    *,
    device,
    batch_size: int,
    strategies: Sequence[PublicNodeStrategy],
) -> dict[str, list[FramePeaks]]:
    import torch
    import zarr

    array = zarr.open_group(str(sample_path), mode="r")["0"]
    public_by_frame = graph_points_by_frame(public_graph)
    frame_count = int(array.shape[0])
    results = {strategy.name: [] for strategy in strategies}
    for start in range(0, frame_count, batch_size):
        frames = list(range(start, min(start + batch_size, frame_count)))
        loaded = [
            normalize_spatialdino_frame(
                array[frame, :, ::4, ::4].astype(np.float32)
            )
            for frame in frames
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = predict_probability_batch(
            model, images, yx_tta=True
        ).cpu().numpy()
        for local_index, frame in enumerate(frames):
            points = np.asarray(
                public_by_frame.get(frame, np.empty((0, 3))), dtype=np.float32
            ).reshape(-1, 3) / SPATIAL_DOWNSAMPLE.astype(np.float32)
            confidence = np.ones(len(points), dtype=np.float32)
            for strategy in strategies:
                refined = refine_public_points(
                    probabilities[local_index, 0], points, strategy
                )
                results[strategy.name].append(
                    FramePeaks(frame, refined, confidence)
                )
        del images, probabilities
    return results


def write_refined_graph(
    public_graph: Path,
    predictions: Sequence[FramePeaks],
    output_path: Path,
) -> dict[str, Any]:
    """Write a topology-identical GEFF with only native coordinates updated."""

    graph = graph_from_geff(public_graph)
    predicted_by_frame = {
        int(frame.frame): np.asarray(frame.points_input, dtype=np.float64).reshape(-1, 3)
        for frame in predictions
    }
    consumed = {frame: 0 for frame in predicted_by_frame}
    node_ids: list[int] = []
    native_points: list[np.ndarray] = []
    for row in graph.node_attrs().iter_rows(named=True):
        frame = int(row["t"])
        if frame not in predicted_by_frame:
            raise RuntimeError(f"refined predictions omit graph frame {frame}")
        index = consumed[frame]
        points = predicted_by_frame[frame]
        if index >= len(points):
            raise RuntimeError(f"refined predictions omit nodes in frame {frame}")
        node_ids.append(int(row["node_id"]))
        native_points.append(points[index] * SPATIAL_DOWNSAMPLE)
        consumed[frame] = index + 1
    for frame, points in predicted_by_frame.items():
        if consumed.get(frame, 0) != len(points):
            raise RuntimeError(f"refined predictions add nodes in frame {frame}")
    coordinates = np.asarray(native_points, dtype=np.float64).reshape(-1, 3)
    graph.update_node_attrs(
        attrs={
            "z": coordinates[:, 0].tolist(),
            "y": coordinates[:, 1].tolist(),
            "x": coordinates[:, 2].tolist(),
        },
        node_ids=node_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_geff(output_path)
    return {
        "path": str(output_path),
        "nodes": int(graph.num_nodes()),
        "edges": int(graph.num_edges()),
        "topology_preserved": True,
    }


def evaluate_movies(
    model,
    competition_dir: Path,
    public_predictions: Path,
    stems: Sequence[str],
    *,
    device,
    batch_size: int,
    strategies: Sequence[PublicNodeStrategy],
    partial_path: Path,
    started: float,
    max_wall_seconds: float,
    output_graph_dir: Path | None = None,
    output_graph_strategy: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    import zarr

    rows = {strategy.name: [] for strategy in strategies}
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        public_graph = public_predictions / f"{stem}.geff"
        if not public_graph.exists():
            raise FileNotFoundError(public_graph)
        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        truth = graph_points_by_frame(truth_path)
        predictions = predict_strategies(
            model,
            sample_path,
            public_graph,
            device=device,
            batch_size=batch_size,
            strategies=strategies,
        )
        expected_nodes = sum(len(value) for value in graph_points_by_frame(public_graph).values())
        for strategy in strategies:
            score = score_predictions(predictions[strategy.name], truth, 0.5)
            if int(score["predicted_nodes"]) != expected_nodes:
                raise RuntimeError("public-node refinement changed node count")
            rows[strategy.name].append(
                {
                    "stem": stem,
                    "candidate": score,
                    "public_nodes": expected_nodes,
                    "frame_count": frame_count,
                }
            )
            if output_graph_dir is not None and strategy.name == output_graph_strategy:
                rows[strategy.name][-1]["refined_graph"] = write_refined_graph(
                    public_graph,
                    predictions[strategy.name],
                    output_graph_dir / f"{stem}.geff",
                )
        partial_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "PUBLIC NODE REFINEMENT EVAL",
            json.dumps(
                {
                    "stem": stem,
                    "strategies": {
                        name: values[-1]["candidate"] for name, values in rows.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("public-node refinement reached its wall guard")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--public-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("public-node refinement requires CUDA")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("status") != "completed" or training.get("validation_overlap") != []:
        raise ValueError("detector training evidence is invalid")
    if training.get("public_predictions_copied") is not False:
        raise ValueError("independent detector provenance is invalid")
    if sha256_file(args.model_path) != training.get("best_weight_sha256"):
        raise ValueError("detector checkpoint hash mismatch")

    try:
        from lsm_fm_image_text_model import build_lsm_fm_detector
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import build_lsm_fm_detector

    model = build_lsm_fm_detector(
        args.lsm_fm_checkpoint,
        expected_sha256=args.lsm_fm_checkpoint_sha256,
    )
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected detector parameter count")
    device = torch.device("cuda")
    model.requires_grad_(False).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    selection_rows = evaluate_movies(
        model,
        args.competition_dir,
        args.public_predictions,
        SCREEN_STEMS,
        device=device,
        batch_size=args.batch_size,
        strategies=STRATEGIES,
        partial_path=args.output_dir / "selection_public_node_refinement_partial.json",
        started=started,
        max_wall_seconds=args.max_wall_seconds,
    )
    selection = {name: summarize_rows(rows) for name, rows in selection_rows.items()}
    selected, diagnostics = select_public_node_strategy(selection)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "selection": selection,
        "selection_diagnostics": diagnostics,
        "selected_strategy": selected,
        "selection_passed": selected is not None,
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "competition_submission_performed": False,
        "public_leaderboard_used_for_selection": False,
        "public_graph_topology_preserved": True,
        "provenance": {
            "detector_checkpoint_sha256": sha256_file(args.model_path),
            "detector_training_result_sha256": sha256_file(args.training_result),
            "detector_parameters": EXPECTED_DETECTOR_PARAMETERS,
            "coordinate_source": "independent_lsm_fm_probability",
            "node_ids_preserved": True,
            "node_count_preserved": True,
            "edges_preserved": True,
        },
    }
    if selected is not None:
        strategy = next(value for value in STRATEGIES if value.name == selected)
        acceptance_rows = evaluate_movies(
            model,
            args.competition_dir,
            args.public_predictions,
            ACCEPTANCE_STEMS,
            device=device,
            batch_size=args.batch_size,
            strategies=(PublicNodeStrategy(CONTROL_NAME, 0, 2.0, 0.0), strategy),
            partial_path=args.output_dir / "acceptance_public_node_refinement_partial.json",
            started=started,
            max_wall_seconds=args.max_wall_seconds,
            output_graph_dir=args.output_dir / "refined_acceptance_graphs",
            output_graph_strategy=selected,
        )
        acceptance = {
            name: summarize_rows(rows) for name, rows in acceptance_rows.items()
        }
        control = acceptance[CONTROL_NAME]
        candidate = acceptance[selected]
        per_movie_delta = [
            float(cand["candidate"]["annotated_node_recall"])
            - float(base["candidate"]["annotated_node_recall"])
            for cand, base in zip(candidate["rows"], control["rows"])
        ]
        result.update(
            {
                "acceptance": acceptance,
                "acceptance_opened": True,
                "promotion_passed": (
                    int(candidate["matched_gt_nodes"]) >= int(control["matched_gt_nodes"])
                    and min(per_movie_delta) >= -MAXIMUM_MOVIE_RECALL_REGRESSION
                ),
                "acceptance_minimum_movie_recall_delta": min(per_movie_delta),
            }
        )
    output = args.output_dir / "lsm_fm_public_node_refinement.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PUBLIC NODE REFINEMENT COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
