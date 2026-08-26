from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from correction import SwapConfig, appearance_pair_swaps
    from encoder import load_spatialdino_vits8, sha256_file
    from appearance import sample_patch_embeddings
    from train_biohub_hoct_probe import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        _score_partition,
        atomic_json,
        read_graph_video,
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
    )
except ModuleNotFoundError:
    from research.hoct_graph.train_biohub_hoct_probe import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        _score_partition,
        atomic_json,
        read_graph_video,
    )
    from research.spatialdino_association.appearance import sample_patch_embeddings
    from research.spatialdino_association.correction import (
        SwapConfig,
        appearance_pair_swaps,
    )
    from research.spatialdino_association.encoder import (
        load_spatialdino_vits8,
        sha256_file,
    )
    from research.trackastra_graph.rerank_submission import (
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
    )


CHECKPOINT_SHA256 = "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
ENCODER_SHAPE = (64, 64, 64)


def normalize_frame(volume: np.ndarray) -> np.ndarray:
    image = np.asarray(volume, dtype=np.float32)
    if image.shape != ENCODER_SHAPE:
        raise ValueError(f"expected a 64-cube encoder frame, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image frame contains non-finite voxels")
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        raise ValueError("image frame has no intensity range")
    return (image - minimum) / (maximum - minimum)


def selection_configurations() -> list[SwapConfig]:
    return [
        SwapConfig(
            min_appearance_gain=gain,
            max_pair_distance_um=12.0,
            max_total_distance_increase_um=distance_increase,
            base_lock_probability=lock,
            fallback_edge_probability=0.80,
        )
        for gain in (0.02, 0.05, 0.10)
        for distance_increase in (0.0, 1.0, 2.0)
        for lock in (0.90, 0.95)
    ]


def robust_selection_key(row: dict[str, Any]) -> tuple[float, float, float, int, float]:
    summary = row["selection_summary"]
    return (
        float(row["selection_min_delta_vs_base"]),
        float(summary["proxy_score"]),
        float(summary["worst_movie"]),
        -int(row["total_swaps"]),
        float(row["config"]["min_appearance_gain"]),
    )


def _compact_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "selected"}


def correct_movie(video, embeddings: np.ndarray, config: SwapConfig):
    return appearance_pair_swaps(
        video.node_ids,
        video.times,
        video.coords_voxel,
        embeddings,
        video.edges,
        config=config,
        edge_probabilities=video.edge_probabilities,
        voxel_scale_um=VOXEL_SCALE_UM,
    )


@torch.inference_mode()
def extract_movie_embeddings(
    model: torch.nn.Module,
    video,
    sample_path: Path,
    device: torch.device,
    *,
    batch_size: int = 4,
    deadline: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import zarr

    array = zarr.open_group(str(sample_path), mode="r")["0"]
    if tuple(array.shape[1:]) != (64, 256, 256):
        raise ValueError(f"unexpected Biohub image shape for {video.stem}: {array.shape}")
    frames = sorted(int(value) for value in np.unique(video.times))
    if not frames or frames[0] < 0 or frames[-1] >= int(array.shape[0]):
        raise ValueError(f"node times are outside the image for {video.stem}")
    output = np.zeros((len(video.node_ids), model.embed_dim), dtype=np.float32)
    frame_seconds: list[float] = []
    for start in range(0, len(frames), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"SpatialDINO deadline reached during {video.stem}")
        frame_batch = frames[start : start + batch_size]
        loaded = [
            normalize_frame(np.asarray(array[frame, :, ::4, ::4]))
            for frame in frame_batch
        ]
        tensor = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        batch_started = time.monotonic()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            feature_batch = model(tensor)
        feature_batch = feature_batch.float().cpu()
        frame_seconds.append((time.monotonic() - batch_started) / len(frame_batch))
        for batch_index, frame in enumerate(frame_batch):
            rows = np.flatnonzero(video.times == frame)
            encoder_coords = video.coords_voxel[rows].astype(np.float32, copy=True)
            encoder_coords[:, 1:] /= 4.0
            encoder_coords = np.clip(
                encoder_coords,
                0.0,
                np.asarray(ENCODER_SHAPE, dtype=np.float32) - 1.0,
            )
            output[rows] = sample_patch_embeddings(
                feature_batch[batch_index],
                encoder_coords,
                input_shape_zyx=ENCODER_SHAPE,
            )
        del tensor, feature_batch
    if not np.isfinite(output).all():
        raise RuntimeError(f"SpatialDINO produced non-finite embeddings for {video.stem}")
    norms = np.linalg.norm(output, axis=1)
    return output, {
        "stem": video.stem,
        "frames": len(frames),
        "nodes": len(video.node_ids),
        "embedding_dimension": output.shape[1],
        "mean_embedding_norm": float(norms.mean()),
        "mean_encoder_seconds_per_frame": float(np.mean(frame_seconds)),
        "image_downsample_zyx": [1, 4, 4],
    }


def validate(
    checkpoint: Path,
    competition_dir: Path,
    raw_validation_root: Path,
    processed_validation_csv: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_wall_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + max_wall_seconds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for SpatialDINO validation")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_spatialdino_vits8(
        checkpoint,
        expected_sha256=CHECKPOINT_SHA256,
        map_location="cpu",
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 21_501_312:
        raise RuntimeError("unexpected SpatialDINO parameter count")

    predictions = read_submission(processed_validation_csv)
    if set(predictions) != set(VALIDATION_STEMS):
        raise ValueError(f"processed validation stems mismatch: {sorted(predictions)}")
    raw_videos = read_raw_graphs(raw_validation_root, set(predictions))
    probability_transfer = transfer_raw_edge_probabilities(predictions, raw_videos)
    train_dir = competition_dir / "train"

    selection_embeddings: dict[str, np.ndarray] = {}
    extraction: dict[str, dict[str, Any]] = {}
    for stem in SELECTION_STEMS:
        print(f"SPATIALDINO CLEAN SELECTION INFERENCE: {stem}", flush=True)
        selection_embeddings[stem], extraction[stem] = extract_movie_embeddings(
            model,
            predictions[stem],
            train_dir / f"{stem}.zarr",
            device,
            batch_size=batch_size,
            deadline=deadline,
        )

    selection_truths = {
        stem: read_graph_video(train_dir / f"{stem}.geff") for stem in SELECTION_STEMS
    }
    base_selection_edges = {
        stem: [tuple(map(int, edge)) for edge in predictions[stem].edges.tolist()]
        for stem in SELECTION_STEMS
    }
    base_selection_summary = _score_partition(
        predictions,
        selection_truths,
        train_dir,
        base_selection_edges,
        SELECTION_STEMS,
    )
    base_selection_by_stem = {
        stem: _score_partition(
            predictions,
            selection_truths,
            train_dir,
            base_selection_edges,
            (stem,),
        )
        for stem in SELECTION_STEMS
    }

    configurations: list[dict[str, Any]] = []
    for config in selection_configurations():
        edges_by_stem: dict[str, list[tuple[int, int]]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for stem in SELECTION_STEMS:
            edges_by_stem[stem], full_diagnostics = correct_movie(
                predictions[stem], selection_embeddings[stem], config
            )
            diagnostics[stem] = _compact_diagnostics(full_diagnostics)
        by_stem = {
            stem: _score_partition(
                predictions, selection_truths, train_dir, edges_by_stem, (stem,)
            )
            for stem in SELECTION_STEMS
        }
        deltas = {
            stem: by_stem[stem]["proxy_score"]
            - base_selection_by_stem[stem]["proxy_score"]
            for stem in SELECTION_STEMS
        }
        configurations.append(
            {
                "config": config.to_dict(),
                "selection_summary": _score_partition(
                    predictions,
                    selection_truths,
                    train_dir,
                    edges_by_stem,
                    SELECTION_STEMS,
                ),
                "selection_by_stem": by_stem,
                "selection_delta_by_stem": deltas,
                "selection_min_delta_vs_base": min(deltas.values()),
                "total_swaps": sum(row["selected_swaps"] for row in diagnostics.values()),
                "diagnostics": diagnostics,
            }
        )
    selected = max(configurations, key=robust_selection_key)
    selection_delta = (
        selected["selection_summary"]["proxy_score"]
        - base_selection_summary["proxy_score"]
    )
    selection_passed = bool(
        selected["total_swaps"] > 0
        and selection_delta > 0
        and selected["selection_min_delta_vs_base"] >= 0
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "model_family": "SpatialDINO frozen ViT-S/8 appearance correction",
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": 21_501_312,
        "prediction_graph_kind": "degree-preserving pair swaps on frozen processed comparator",
        "selection_stems": list(SELECTION_STEMS),
        "acceptance_stems": list(ACCEPTANCE_STEMS),
        "base_selection_summary": base_selection_summary,
        "base_selection_by_stem": base_selection_by_stem,
        "selected": selected,
        "selection_delta_vs_base": selection_delta,
        "selection_passed": selection_passed,
        "extraction": extraction,
        "edge_probability_transfer": probability_transfer,
        "configurations": configurations,
        "acceptance_ground_truth_loaded_after_selection_freeze": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if not selection_passed:
        payload.update(
            {
                "acceptance_skipped": True,
                "acceptance_passed": False,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_json(output_dir / "complete_movie_validation.json", payload)
        return payload

    frozen_config = SwapConfig(**selected["config"])
    acceptance_embeddings: dict[str, np.ndarray] = {}
    for stem in ACCEPTANCE_STEMS:
        print(f"SPATIALDINO CLEAN ACCEPTANCE INFERENCE: {stem}", flush=True)
        acceptance_embeddings[stem], extraction[stem] = extract_movie_embeddings(
            model,
            predictions[stem],
            train_dir / f"{stem}.zarr",
            device,
            batch_size=batch_size,
            deadline=deadline,
        )
    acceptance_edges: dict[str, list[tuple[int, int]]] = {}
    acceptance_diagnostics: dict[str, dict[str, Any]] = {}
    for stem in ACCEPTANCE_STEMS:
        acceptance_edges[stem], diagnostics = correct_movie(
            predictions[stem], acceptance_embeddings[stem], frozen_config
        )
        acceptance_diagnostics[stem] = _compact_diagnostics(diagnostics)

    # Acceptance labels are first opened after the entire correction config is frozen.
    acceptance_truths = {
        stem: read_graph_video(train_dir / f"{stem}.geff") for stem in ACCEPTANCE_STEMS
    }
    base_acceptance_edges = {
        stem: [tuple(map(int, edge)) for edge in predictions[stem].edges.tolist()]
        for stem in ACCEPTANCE_STEMS
    }
    base_acceptance_summary = _score_partition(
        predictions,
        acceptance_truths,
        train_dir,
        base_acceptance_edges,
        ACCEPTANCE_STEMS,
    )
    acceptance_summary = _score_partition(
        predictions,
        acceptance_truths,
        train_dir,
        acceptance_edges,
        ACCEPTANCE_STEMS,
    )
    base_acceptance_by_stem = {
        stem: _score_partition(
            predictions,
            acceptance_truths,
            train_dir,
            base_acceptance_edges,
            (stem,),
        )
        for stem in ACCEPTANCE_STEMS
    }
    acceptance_by_stem = {
        stem: _score_partition(
            predictions, acceptance_truths, train_dir, acceptance_edges, (stem,)
        )
        for stem in ACCEPTANCE_STEMS
    }
    acceptance_delta_by_stem = {
        stem: acceptance_by_stem[stem]["proxy_score"]
        - base_acceptance_by_stem[stem]["proxy_score"]
        for stem in ACCEPTANCE_STEMS
    }
    acceptance_delta = (
        acceptance_summary["proxy_score"] - base_acceptance_summary["proxy_score"]
    )
    acceptance_swaps = sum(
        row["selected_swaps"] for row in acceptance_diagnostics.values()
    )
    acceptance_passed = bool(
        acceptance_swaps > 0
        and acceptance_delta > 0
        and min(acceptance_delta_by_stem.values()) >= -0.005
        and acceptance_summary["worst_movie"]
        >= base_acceptance_summary["worst_movie"] - 0.005
    )
    payload.update(
        {
            "acceptance_skipped": False,
            "acceptance_ground_truth_loaded_after_selection_freeze": True,
            "base_acceptance_summary": base_acceptance_summary,
            "base_acceptance_by_stem": base_acceptance_by_stem,
            "acceptance_summary": acceptance_summary,
            "acceptance_by_stem": acceptance_by_stem,
            "acceptance_delta_by_stem": acceptance_delta_by_stem,
            "acceptance_delta_vs_base": acceptance_delta,
            "acceptance_diagnostics": acceptance_diagnostics,
            "acceptance_swaps": acceptance_swaps,
            "acceptance_passed": acceptance_passed,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    atomic_json(output_dir / "complete_movie_validation.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--raw-validation-root", type=Path, required=True)
    parser.add_argument("--processed-validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-wall-seconds", type=float, default=6600)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = validate(
        args.checkpoint,
        args.competition_dir,
        args.raw_validation_root,
        args.processed_validation_csv,
        args.output_dir,
        batch_size=args.batch_size,
        max_wall_seconds=args.max_wall_seconds,
    )
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": result["elapsed_seconds"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "selection_delta_vs_base": result["selection_delta_vs_base"],
        "selection_passed": result["selection_passed"],
        "acceptance_skipped": result["acceptance_skipped"],
        "acceptance_delta_vs_base": result.get("acceptance_delta_vs_base"),
        "association_acceptance_passed": result["acceptance_passed"],
        "complete_movie_validation_sha256": sha256_file(
            args.output_dir / "complete_movie_validation.json"
        ),
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
