#!/usr/bin/env python
"""Score a synthetic-gated localization ensemble on frozen real probe frames.

The four movies were opened by earlier detector experiments, so this is a
development transfer check rather than final acceptance.  The committed
consensus policy is applied without threshold, member-subset, or blend search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_localization.consensus import NATIVE_VOXEL_UM, select_consensus_offsets
from research.temporal_localization.inference import MovieGraphArrays, predict_member
from research.temporal_localization.model import EXPECTED_PARAMETER_COUNT, TemporalNodeLocalizationModel
from research.temporal_localization.train_synthetic_localizer import RUN_ID as TRAINING_RUN_ID


RUN_ID = "temporal-node-localizer-real-development-v1"
STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
MATCH_RADIUS_UM = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_probe_cache(root: Path) -> dict[str, Any]:
    manifest_path = root / "probe_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-division-probe-frame-cache-v1"
        and manifest.get("summary") == {"bytes": 66_682_915, "files": 23, "frames": 15, "movies": 4}
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
    ):
        raise ValueError("real development frame-cache manifest changed")
    for row in manifest["files"]:
        path = root / row["local_relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"real development cache file changed: {row['local_relative_path']}")
    return manifest


def graph_arrays(path: Path) -> MovieGraphArrays:
    import tracksdata as td

    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = graph.node_attrs().select("node_id", "t", "z", "y", "x").to_dicts()
    edges = graph.edge_attrs().select("source_id", "target_id").to_dicts()
    return MovieGraphArrays(
        node_ids=np.asarray([row["node_id"] for row in nodes], dtype=np.int64),
        times=np.asarray([row["t"] for row in nodes], dtype=np.int64),
        coordinates_zyx_voxel=np.asarray(
            [[row["z"], row["y"], row["x"]] for row in nodes], dtype=np.float32
        ).reshape(-1, 3),
        edges_by_node_id=np.asarray(
            [[row["source_id"], row["target_id"]] for row in edges], dtype=np.int64
        ).reshape(-1, 2),
    )


def cached_interior_frames(movie_root: Path) -> tuple[int, ...]:
    chunk_root = movie_root / "0" / "c"
    frames = {int(path.name) for path in chunk_root.iterdir() if path.is_dir()}
    centers = tuple(sorted(frame for frame in frames if {frame - 1, frame, frame + 1} <= frames))
    if not centers:
        raise ValueError(f"movie cache has no complete temporal triplet: {movie_root.name}")
    return centers


def frame_match(predicted_voxel: np.ndarray, truth_voxel: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted_voxel, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth_voxel, dtype=np.float64).reshape(-1, 3)
    if not len(predicted) or not len(truth):
        return {"predicted": len(predicted), "truth": len(truth), "matched": 0, "distances_um": []}
    distances = np.linalg.norm(
        predicted[:, None] * NATIVE_VOXEL_UM[None, None]
        - truth[None] * NATIVE_VOXEL_UM[None, None],
        axis=2,
    )
    rows, columns = linear_sum_assignment(distances)
    accepted = [float(distances[row, column]) for row, column in zip(rows, columns, strict=True) if distances[row, column] <= MATCH_RADIUS_UM]
    return {
        "predicted": len(predicted),
        "truth": len(truth),
        "matched": len(accepted),
        "distances_um": accepted,
    }


def summarize_matches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [value for row in rows for value in row["distances_um"]]
    truth = sum(int(row["truth"]) for row in rows)
    matched = sum(int(row["matched"]) for row in rows)
    return {
        "frames": len(rows),
        "predicted_nodes": sum(int(row["predicted"]) for row in rows),
        "annotated_nodes": truth,
        "matched_nodes": matched,
        "annotated_node_recall": matched / truth if truth else 0.0,
        "mean_matched_distance_um": float(np.mean(distances)) if distances else None,
        "p90_matched_distance_um": float(np.quantile(distances, 0.9)) if distances else None,
    }


def development_gate(movies: list[dict[str, Any]]) -> dict[str, Any]:
    no_movie_recall_regression = all(
        movie["candidate"]["matched_nodes"] >= movie["baseline"]["matched_nodes"]
        for movie in movies
    )
    baseline_matched = sum(movie["baseline"]["matched_nodes"] for movie in movies)
    candidate_matched = sum(movie["candidate"]["matched_nodes"] for movie in movies)
    baseline_distance_values = [
        movie["baseline"]["mean_matched_distance_um"]
        for movie in movies
        if movie["baseline"]["mean_matched_distance_um"] is not None
    ]
    candidate_distance_values = [
        movie["candidate"]["mean_matched_distance_um"]
        for movie in movies
        if movie["candidate"]["mean_matched_distance_um"] is not None
    ]
    mean_distance_improved = bool(
        baseline_distance_values
        and candidate_distance_values
        and float(np.mean(candidate_distance_values)) < float(np.mean(baseline_distance_values))
    )
    passed = bool(
        no_movie_recall_regression
        and candidate_matched > baseline_matched
        and mean_distance_improved
        and all(movie["consensus"]["global_gate_passed"] for movie in movies)
    )
    return {
        "passed": passed,
        "no_movie_recall_regression": no_movie_recall_regression,
        "matched_node_gain": candidate_matched - baseline_matched,
        "mean_movie_matched_distance_improved": mean_distance_improved,
        "all_global_move_fraction_gates_passed": all(
            movie["consensus"]["global_gate_passed"] for movie in movies
        ),
    }


def member_device_index(member_index: int, gpu_count: int) -> int:
    """Assign accepted members deterministically across the visible GPUs."""

    if gpu_count < 1:
        raise RuntimeError("real development probe requires at least one CUDA GPU")
    if member_index < 0:
        raise ValueError("member index must be non-negative")
    return member_index % gpu_count


def load_members(results_root: Path) -> list[tuple[TemporalNodeLocalizationModel, torch.device, dict[str, Any]]]:
    terminals = sorted(results_root.glob("gpu_*/member_*/worker_terminal.json"))
    accepted: list[tuple[TemporalNodeLocalizationModel, torch.device, dict[str, Any]]] = []
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("real development probe requires at least one CUDA GPU")
    for terminal_path in terminals:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        checkpoint = terminal_path.parent / "localization_model.pt"
        if not (
            terminal.get("status") == "completed"
            and terminal.get("run_id") == TRAINING_RUN_ID
            and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and terminal.get("selection_gate_passed") is True
            and terminal.get("audit_gate_passed") is True
            and terminal.get("division_critical_selection_gate_passed") is True
            and terminal.get("division_critical_audit_gate_passed") is True
            and terminal.get("real_selection_gate_passed") is True
            and terminal.get("real_audit_gate_passed") is True
            and terminal.get("real_division_critical_selection_gate_passed") is True
            and terminal.get("real_division_critical_audit_gate_passed") is True
            and terminal.get("serialized_checkpoint_selection_gate_passed") is True
            and terminal.get("checkpoint_frozen_before_audit") is True
            and terminal.get("model_sha256") == sha256_file(checkpoint)
            and terminal.get("real_replay_probability") == 0.25
            and terminal.get("competition_train_data_read") is True
            and terminal.get("competition_test_data_read") is False
            and terminal.get("public_code_copied") is False
            and terminal.get("public_predictions_copied") is False
            and terminal.get("public_leaderboard_used_for_selection") is False
            and terminal.get("submission_created") is False
        ):
            continue
        device = torch.device(f"cuda:{member_device_index(len(accepted), gpu_count)}")
        model = TemporalNodeLocalizationModel().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval().requires_grad_(False)
        accepted.append((model, device, terminal))
    if len(accepted) < 3:
        raise RuntimeError("fewer than three independently accepted localization members")
    return accepted


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_probe_cache(args.probe_root)
    members = load_members(args.results_root)
    import zarr

    movie_results: list[dict[str, Any]] = []
    for stem in STEMS:
        movie_path = args.probe_root / "train" / f"{stem}.zarr"
        array = zarr.open_group(str(movie_path), mode="r")["0"]
        frames = cached_interior_frames(movie_path)
        control = graph_arrays(args.control_root / f"{stem}.geff")
        truth = graph_arrays(args.truth_root / f"{stem}.geff")
        predictions = [
            predict_member(
                model,
                control,
                lambda frame, values=array: np.asarray(values[frame]),
                frames,
                device=device,
                voxel_size_zyx_um=NATIVE_VOXEL_UM,
                spatial_shape_zyx=tuple(int(value) for value in array.shape[1:]),
                timepoints=int(array.shape[0]),
                batch_size=args.batch_size,
            )
            for model, device, _terminal in members
        ]
        row_inventory = predictions[0][0]
        if any(not np.array_equal(rows, row_inventory) for rows, _offset, _sigma, _safe in predictions[1:]):
            raise RuntimeError("localization members returned different node inventories")
        consensus = select_consensus_offsets(
            np.stack([row[1] for row in predictions]),
            np.stack([row[2] for row in predictions]),
            np.stack([row[3] for row in predictions]),
        )
        candidate_coords = control.coordinates_zyx_voxel.copy()
        candidate_coords[row_inventory] += consensus.offsets_um / NATIVE_VOXEL_UM[None]
        baseline_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for frame in frames:
            control_rows = np.flatnonzero(control.times == frame)
            truth_rows = np.flatnonzero(truth.times == frame)
            baseline_rows.append(frame_match(control.coordinates_zyx_voxel[control_rows], truth.coordinates_zyx_voxel[truth_rows]))
            candidate_rows.append(frame_match(candidate_coords[control_rows], truth.coordinates_zyx_voxel[truth_rows]))
        movie_results.append(
            {
                "stem": stem,
                "frames": list(frames),
                "baseline": summarize_matches(baseline_rows),
                "candidate": summarize_matches(candidate_rows),
                "consensus": consensus.report,
            }
        )
    gate = development_gate(movie_results)
    return {
        "schema_version": 1,
        "status": "development_passed" if gate["passed"] else "development_rejected",
        "run_id": RUN_ID,
        "training_run_id": TRAINING_RUN_ID,
        "members": [
            {"seed": terminal["seed"], "model_sha256": terminal["model_sha256"]}
            for _model, _device, terminal in members
        ],
        "inference_gpu_count": torch.cuda.device_count(),
        "inference_device_policy": "accepted_members_round_robin_across_visible_cuda_devices",
        "movies": movie_results,
        "gate": gate,
        "acceptance_labels_already_opened": True,
        "development_only": True,
        "policy_or_member_selection_performed": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("development batch size must be positive")
    result = run(args)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
