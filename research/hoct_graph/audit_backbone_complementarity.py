from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from biohub_adapter import iter_pair_tiles, predict_window
    from rerank_submission import read_submission
    from trainer import read_graph_video
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.hoct_graph.biohub_adapter import iter_pair_tiles, predict_window
    from research.trackastra_graph.rerank_submission import read_submission
    from research.trackastra_graph.train_biohub_graph_transformer import (
        read_graph_video,
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or float(first.std()) == 0 or float(second.std()) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def audit_backbones(
    graph_path: Path,
    general_model_path: Path,
    ctc_model_path: Path,
    *,
    device: torch.device,
    max_tiles: int,
    movie_stem: str | None = None,
) -> dict[str, object]:
    if graph_path.suffix.casefold() == ".csv":
        videos = read_submission(graph_path)
        if movie_stem is None:
            if len(videos) != 1:
                raise ValueError("movie-stem is required for a multi-movie CSV")
            video = next(iter(videos.values()))
        else:
            video = videos[movie_stem]
    else:
        video = read_graph_video(graph_path)
    general = torch.jit.load(str(general_model_path), map_location=device).eval()
    ctc = torch.jit.load(str(ctc_model_path), map_location=device).eval()

    general_probabilities: list[np.ndarray] = []
    ctc_probabilities: list[np.ndarray] = []
    general_logits: list[np.ndarray] = []
    ctc_logits: list[np.ndarray] = []
    top_parent_agreements = 0
    target_groups = 0
    tiles_seen = 0

    source_times = sorted(
        source_t
        for source_t in video.ids_by_time
        if source_t + 1 in video.ids_by_time
    )
    # Spread the audit across the movie instead of measuring only its first
    # developmental state.
    sampled_indices = np.linspace(
        0, max(len(source_times) - 1, 0), min(len(source_times), max_tiles), dtype=int
    )
    for source_t in (source_times[index] for index in sampled_indices.tolist()):
        tiles = iter_pair_tiles(
            video.node_ids,
            video.times,
            video.coords_voxel,
            source_t=source_t,
        )
        for tile in tiles:
            if tiles_seen >= max_tiles:
                break
            general_prediction = predict_window(general, tile.window, device=device)
            ctc_prediction = predict_window(ctc, tile.window, device=device)
            owned = np.flatnonzero(tile.core_edge_mask)
            if len(owned) == 0:
                continue
            general_probabilities.append(general_prediction.edge_probabilities[owned])
            ctc_probabilities.append(ctc_prediction.edge_probabilities[owned])
            general_logits.append(general_prediction.edge_logits[owned])
            ctc_logits.append(ctc_prediction.edge_logits[owned])

            targets = tile.window.edge_indices[owned, 1]
            for target in np.unique(targets):
                local = owned[targets == target]
                if len(local) == 0:
                    continue
                general_parent = int(
                    tile.window.edge_indices[
                        local[np.argmax(general_prediction.edge_probabilities[local])], 0
                    ]
                )
                ctc_parent = int(
                    tile.window.edge_indices[
                        local[np.argmax(ctc_prediction.edge_probabilities[local])], 0
                    ]
                )
                top_parent_agreements += int(general_parent == ctc_parent)
                target_groups += 1
            tiles_seen += 1
        if tiles_seen >= max_tiles:
            break

    if not general_probabilities:
        raise RuntimeError("No owned HOCT candidate edges were audited")
    general_probability = np.concatenate(general_probabilities)
    ctc_probability = np.concatenate(ctc_probabilities)
    general_logit = np.concatenate(general_logits)
    ctc_logit = np.concatenate(ctc_logits)
    return {
        "schema_version": 1,
        "graph": graph_path.name,
        "movie_stem": video.stem,
        "device": str(device),
        "tiles": tiles_seen,
        "owned_candidate_edges": len(general_probability),
        "target_parent_groups": target_groups,
        "general_probability_median": float(np.median(general_probability)),
        "ctc_probability_median": float(np.median(ctc_probability)),
        "general_probability_p95": float(np.quantile(general_probability, 0.95)),
        "ctc_probability_p95": float(np.quantile(ctc_probability, 0.95)),
        "probability_pearson": _safe_correlation(
            general_probability, ctc_probability
        ),
        "logit_pearson": _safe_correlation(general_logit, ctc_logit),
        "probability_mean_absolute_difference": float(
            np.mean(np.abs(general_probability - ctc_probability))
        ),
        "top_parent_agreement": top_parent_agreements / max(target_groups, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--general-model", type=Path, required=True)
    parser.add_argument("--ctc-model", type=Path, required=True)
    parser.add_argument("--movie-stem")
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_tiles <= 0:
        raise ValueError("max-tiles must be positive")
    result = audit_backbones(
        args.graph,
        args.general_model,
        args.ctc_model,
        device=torch.device(args.device),
        max_tiles=args.max_tiles,
        movie_stem=args.movie_stem,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
