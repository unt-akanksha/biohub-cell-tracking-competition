from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from biohub_adapter import (
        HOCTTile,
        build_window,
        iter_pair_tiles,
        parental_softmax,
        predict_window,
    )
    from hybrid_linker import HybridLinkConfig
    from rerank_submission import (
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
    )
    from trainer import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        GraphVideo,
        aggregate_scores,
        atomic_json,
        hybrid_link_movie,
        link_movie,
        read_graph_video,
        read_true_node_count,
        score_linked_video,
        summarize_stored_edge_probabilities,
    )
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.hoct_graph.biohub_adapter import (
        HOCTTile,
        build_window,
        iter_pair_tiles,
        parental_softmax,
        predict_window,
    )
    from research.trackastra_graph.hybrid_linker import HybridLinkConfig
    from research.trackastra_graph.rerank_submission import (
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
    )
    from research.trackastra_graph.train_biohub_graph_transformer import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        GraphVideo,
        aggregate_scores,
        atomic_json,
        hybrid_link_movie,
        link_movie,
        read_graph_video,
        read_true_node_count,
        score_linked_video,
        summarize_stored_edge_probabilities,
    )


@dataclass(frozen=True)
class ProbeConfig:
    window_size: int = 5
    window_stride: int = 4
    neighbors: int = 8
    max_distance: float = 80.0
    negative_ratio: float = 3.0
    hard_negative_fraction: float = 0.6
    epochs: int = 12
    batch_size: int = 8192
    learning_rate: float = 2e-3
    anchor_weight: float = 1e-4
    weight_decay: float = 1e-5
    max_examples: int = 600_000
    seed: int = 20260826


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def covering_window_starts(
    times: np.ndarray, window_size: int = 5, stride: int = 4
) -> tuple[int, ...]:
    """Return windows that cover every consecutive transition at least once."""
    values = np.asarray(times, dtype=np.int32).reshape(-1)
    if len(values) == 0:
        return ()
    if window_size < 2 or stride <= 0 or stride > window_size - 1:
        raise ValueError("Window stride must be in [1, window_size - 1]")
    minimum = int(values.min())
    maximum = int(values.max())
    starts = list(range(minimum, maximum, stride))
    if starts and starts[-1] + window_size - 1 < maximum:
        starts.append(maximum - window_size + 1)
    return tuple(sorted(set(starts)))


def select_probe_examples(
    features: np.ndarray,
    labels: np.ndarray,
    pretrained_logits: np.ndarray,
    *,
    negative_ratio: float,
    hard_negative_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positives plus a mix of hard and uniformly sampled negatives."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    logits = np.asarray(pretrained_logits, dtype=np.float32).reshape(-1)
    if features.shape[0] != len(labels) or len(logits) != len(labels):
        raise ValueError("Probe example arrays have inconsistent lengths")
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels <= 0.5)
    if len(positive) == 0:
        return np.empty((0, features.shape[1]), np.float32), np.empty(0, np.float32)
    wanted = min(len(negative), int(np.ceil(len(positive) * negative_ratio)))
    hard_count = min(wanted, int(round(wanted * hard_negative_fraction)))
    ranked_negative = negative[np.argsort(logits[negative])[::-1]]
    hard = ranked_negative[:hard_count]
    remainder = ranked_negative[hard_count:]
    random_count = wanted - len(hard)
    uniform = (
        rng.choice(remainder, size=random_count, replace=False)
        if random_count and len(remainder)
        else np.empty(0, dtype=np.int64)
    )
    chosen = np.concatenate([positive, hard, uniform])
    rng.shuffle(chosen)
    return features[chosen], labels[chosen]


def initialize_probe(model: torch.nn.Module, device: torch.device) -> torch.nn.Linear:
    head = torch.nn.Linear(288, 1)
    parameters = dict(model.named_parameters())
    source_weight = parameters.get("head.weight")
    source_bias = parameters.get("head.bias")
    if source_weight is None or source_bias is None:
        raise RuntimeError("Official HOCT checkpoint does not expose its linear head")
    head.weight.data.copy_(source_weight.detach().float().cpu())
    head.bias.data.copy_(source_bias.detach().float().cpu())
    return head.to(device)


def _classification_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    positive = probabilities[labels > 0.5]
    negative = probabilities[labels <= 0.5]
    return {
        "bce": float(F.binary_cross_entropy_with_logits(logits, labels).item()),
        "positive_probability_median": float(positive.median().item()),
        "negative_probability_median": float(negative.median().item()),
        "negative_probability_p95": float(torch.quantile(negative, 0.95).item()),
    }


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    config: ProbeConfig,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    features_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    labels_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32).reshape(-1))
    if len(labels_tensor) == 0 or labels_tensor.sum() == 0:
        raise ValueError("Cannot fit a HOCT probe without positive examples")
    head = initialize_probe(model, device)
    initial_weight = head.weight.detach().clone()
    initial_bias = head.bias.detach().clone()
    with torch.inference_mode():
        before_logits = head(features_tensor[: min(len(features_tensor), 100_000)].to(device)).squeeze(1).cpu()
        before = _classification_summary(before_logits, labels_tensor[: len(before_logits)])

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    n_positive = float(labels_tensor.sum().item())
    n_negative = float(len(labels_tensor) - n_positive)
    positive_weight = torch.tensor(n_negative / max(n_positive, 1.0), device=device)
    epoch_losses: list[float] = []
    head.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(len(features_tensor), generator=generator)
        running_loss = 0.0
        examples = 0
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset : offset + config.batch_size]
            batch_features = features_tensor[indices].to(device, non_blocking=True)
            batch_labels = labels_tensor[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=positive_weight
            )
            anchor = (head.weight - initial_weight).square().mean() + (
                head.bias - initial_bias
            ).square().mean()
            loss = loss + config.anchor_weight * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(indices)
            examples += len(indices)
        epoch_losses.append(running_loss / max(examples, 1))
    head.eval()
    with torch.inference_mode():
        after_logits = head(features_tensor[: min(len(features_tensor), 100_000)].to(device)).squeeze(1).cpu()
        after = _classification_summary(after_logits, labels_tensor[: len(after_logits)])
    report = {
        "examples": len(labels_tensor),
        "positive_examples": int(n_positive),
        "negative_examples": int(n_negative),
        "positive_weight": float(positive_weight.item()),
        "before": before,
        "after": after,
        "epoch_losses": epoch_losses,
        "weight_delta_l2": float((head.weight - initial_weight).norm().item()),
        "bias_delta": float((head.bias - initial_bias).abs().item()),
    }
    return head, report


def extract_training_examples(
    model: torch.nn.Module,
    train_paths: list[Path],
    device: torch.device,
    config: ProbeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    captured_positive_edges: set[tuple[str, int, int]] = set()
    eligible_positive_edges: set[tuple[str, int, int]] = set()
    windows = 0
    candidate_edges_count = 0
    started = time.monotonic()
    for movie_index, path in enumerate(train_paths):
        video = read_graph_video(path)
        true_edges = {(int(source), int(target)) for source, target in video.edges.tolist()}
        for source, target in true_edges:
            if video.time_by_id.get(target) == video.time_by_id.get(source, -2) + 1:
                eligible_positive_edges.add((video.stem, source, target))
        for start in covering_window_starts(
            video.times, config.window_size, config.window_stride
        ):
            window = build_window(
                video.node_ids,
                video.times,
                video.coords_voxel,
                start=start,
                window_size=config.window_size,
                neighbors=config.neighbors,
                max_distance=config.max_distance,
                true_edges=true_edges,
            )
            if len(window.edge_indices) == 0 or window.edge_labels is None:
                continue
            prediction = predict_window(model, window, device=device)
            selected_features, selected_labels = select_probe_examples(
                prediction.edge_features,
                window.edge_labels,
                prediction.edge_logits,
                negative_ratio=config.negative_ratio,
                hard_negative_fraction=config.hard_negative_fraction,
                rng=rng,
            )
            if len(selected_labels) == 0:
                continue
            feature_chunks.append(selected_features)
            label_chunks.append(selected_labels)
            candidate_edges_count += len(window.edge_indices)
            windows += 1
            for edge_index in np.flatnonzero(window.edge_labels > 0.5):
                source_local, target_local = window.edge_indices[edge_index]
                captured_positive_edges.add(
                    (
                        video.stem,
                        int(window.node_ids[source_local]),
                        int(window.node_ids[target_local]),
                    )
                )
        print(
            f"HOCT FEATURE EXTRACTION {movie_index + 1}/{len(train_paths)}: "
            f"{video.stem} examples={sum(len(chunk) for chunk in label_chunks)}",
            flush=True,
        )
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    if len(labels) > config.max_examples:
        positive = np.flatnonzero(labels > 0.5)
        negative = np.flatnonzero(labels <= 0.5)
        remaining = max(0, config.max_examples - len(positive))
        chosen_negative = rng.choice(
            negative, size=min(remaining, len(negative)), replace=False
        )
        chosen = np.concatenate([positive, chosen_negative])
        rng.shuffle(chosen)
        features = features[chosen]
        labels = labels[chosen]
    evidence = {
        "movies": len(train_paths),
        "windows": windows,
        "candidate_edges": candidate_edges_count,
        "eligible_consecutive_gt_edges": len(eligible_positive_edges),
        "captured_consecutive_gt_edges": len(captured_positive_edges),
        "candidate_recall": len(captured_positive_edges)
        / max(len(eligible_positive_edges), 1),
        "selected_examples": len(labels),
        "selected_positive_examples": int(labels.sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
    return features, labels, evidence


def _empty_pair_scores(video: GraphVideo) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_t in sorted(video.ids_by_time):
        target_t = source_t + 1
        if target_t not in video.ids_by_time:
            continue
        source_ids = video.ids_by_time[source_t]
        target_ids = video.ids_by_time[target_t]
        scores[source_t] = (
            source_ids,
            target_ids,
            np.zeros((len(source_ids), len(target_ids)), dtype=np.float32),
        )
    return scores


def _scatter_core_probabilities(
    tile: HOCTTile,
    probabilities: np.ndarray,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    window = tile.window
    for edge_index in np.flatnonzero(tile.core_edge_mask):
        source_local, target_local = window.edge_indices[edge_index]
        source_t = int(window.times[source_local])
        source_ids, target_ids, matrix = pair_scores[source_t]
        source_id = int(window.node_ids[source_local])
        target_id = int(window.node_ids[target_local])
        source_row = int(np.searchsorted(source_ids, source_id))
        target_col = int(np.searchsorted(target_ids, target_id))
        # Stable IDs are sorted in Biohub CSV and GEFF inputs.  Fall back to a
        # map if a future artifact changes that harmless ordering detail.
        if source_row >= len(source_ids) or int(source_ids[source_row]) != source_id:
            source_row = {int(value): i for i, value in enumerate(source_ids)}[source_id]
        if target_col >= len(target_ids) or int(target_ids[target_col]) != target_id:
            target_col = {int(value): i for i, value in enumerate(target_ids)}[target_id]
        matrix[source_row, target_col] = max(
            matrix[source_row, target_col], float(probabilities[edge_index])
        )


def predict_movie_variants(
    model: torch.nn.Module,
    probe: torch.nn.Linear,
    video: GraphVideo,
    device: torch.device,
    config: ProbeConfig,
    *,
    core_size: np.ndarray,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    outputs = {
        "pretrained": _empty_pair_scores(video),
        "biohub_probe": _empty_pair_scores(video),
    }
    probe_weight = probe.weight.detach().float().cpu().numpy().reshape(-1)
    probe_bias = float(probe.bias.detach().float().cpu().item())
    for source_t in sorted(outputs["pretrained"]):
        tiles = iter_pair_tiles(
            video.node_ids,
            video.times,
            video.coords_voxel,
            source_t=source_t,
            core_size=core_size,
            context_halo=config.max_distance,
            neighbors=config.neighbors,
            max_distance=config.max_distance,
        )
        for tile in tiles:
            prediction = predict_window(model, tile.window, device=device)
            probe_logits = prediction.edge_features @ probe_weight + probe_bias
            probe_probabilities, _orphan = parental_softmax(
                probe_logits,
                prediction.orphan_logits,
                tile.window.edge_indices,
                tile.window.edge_delta_t,
            )
            _scatter_core_probabilities(
                tile, prediction.edge_probabilities, outputs["pretrained"]
            )
            _scatter_core_probabilities(tile, probe_probabilities, outputs["biohub_probe"])
    return outputs


def _score_partition(
    predictions: dict[str, GraphVideo],
    truths: dict[str, GraphVideo],
    train_dir: Path,
    edges_by_stem: dict[str, list[tuple[int, int]]],
    stems: Iterable[str],
) -> dict[str, Any]:
    return aggregate_scores(
        [
            score_linked_video(
                predictions[stem],
                truths[stem],
                edges_by_stem[stem],
                read_true_node_count(train_dir / f"{stem}.geff"),
            )
            for stem in stems
        ]
    )


def _selection_configurations() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for variant in ("pretrained", "biohub_probe"):
        for edge_threshold in (0.03, 0.06, 0.10, 0.16):
            for division_threshold in (0.08, 0.14, 0.22):
                for division_ratio in (0.35, 0.55):
                    configs.append(
                        {
                            "variant": variant,
                            "method": "hoct_only",
                            "edge_threshold": edge_threshold,
                            "division_threshold": division_threshold,
                            "division_ratio": division_ratio,
                        }
                    )
        for edge_threshold in (0.03, 0.08):
            for base_lock_probability in (0.94, 0.98):
                for base_keep_probability in (0.65, 0.80):
                    for base_bonus in (0.02, 0.05):
                        configs.append(
                            {
                                "variant": variant,
                                "method": "raw_confidence_hybrid",
                                "edge_threshold": edge_threshold,
                                "base_lock_probability": base_lock_probability,
                                "base_keep_probability": base_keep_probability,
                                "base_bonus": base_bonus,
                                "division_threshold": 0.14,
                                "division_ratio": 0.50,
                                "base_division_keep_probability": 0.95,
                            }
                        )
    return configs


def link_configuration(
    video: GraphVideo,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    if config["method"] == "hoct_only":
        return link_movie(
            pair_scores,
            edge_threshold=float(config["edge_threshold"]),
            division_threshold=float(config["division_threshold"]),
            division_ratio=float(config["division_ratio"]),
        )
    hybrid = HybridLinkConfig(
        edge_threshold=float(config["edge_threshold"]),
        base_lock_probability=float(config["base_lock_probability"]),
        base_keep_probability=float(config["base_keep_probability"]),
        base_bonus=float(config["base_bonus"]),
        division_threshold=float(config["division_threshold"]),
        division_ratio=float(config["division_ratio"]),
        base_division_keep_probability=float(config["base_division_keep_probability"]),
    )
    return hybrid_link_movie(
        video, pair_scores, hybrid, use_stored_edge_probabilities=True
    )


def validate_complete_movies(
    model: torch.nn.Module,
    probe: torch.nn.Linear,
    predictions: dict[str, GraphVideo],
    train_dir: Path,
    device: torch.device,
    config: ProbeConfig,
    output_dir: Path,
    core_size: np.ndarray,
) -> dict[str, Any]:
    truths = {
        stem: read_graph_video(train_dir / f"{stem}.geff") for stem in VALIDATION_STEMS
    }
    selection_scores: dict[
        str, dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ] = {}
    for stem in SELECTION_STEMS:
        print(f"HOCT CLEAN SELECTION INFERENCE: {stem}", flush=True)
        selection_scores[stem] = predict_movie_variants(
            model, probe, predictions[stem], device, config, core_size=core_size
        )

    base_edges = {
        stem: [tuple(map(int, edge)) for edge in predictions[stem].edges.tolist()]
        for stem in VALIDATION_STEMS
    }
    base_summary = {
        "selection": _score_partition(
            predictions, truths, train_dir, base_edges, SELECTION_STEMS
        ),
        "acceptance": _score_partition(
            predictions, truths, train_dir, base_edges, ACCEPTANCE_STEMS
        ),
        "all": _score_partition(predictions, truths, train_dir, base_edges, VALIDATION_STEMS),
    }
    configurations: list[dict[str, Any]] = []
    for candidate in _selection_configurations():
        edges = {
            stem: link_configuration(
                predictions[stem],
                selection_scores[stem][candidate["variant"]],
                candidate,
            )
            for stem in SELECTION_STEMS
        }
        configurations.append(
            {
                **candidate,
                "selection_summary": _score_partition(
                    predictions, truths, train_dir, edges, SELECTION_STEMS
                ),
            }
        )
    selected = max(
        configurations,
        key=lambda row: (
            row["selection_summary"]["proxy_score"],
            row["selection_summary"]["worst_movie"],
            -row["selection_summary"]["div_fp"],
            row["method"] == "raw_confidence_hybrid",
        ),
    )

    # Acceptance inference happens only after backbone/head/method/thresholds
    # have been frozen on the two selection movies.
    acceptance_scores: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for stem in ACCEPTANCE_STEMS:
        print(f"HOCT CLEAN ACCEPTANCE INFERENCE: {stem}", flush=True)
        variants = predict_movie_variants(
            model, probe, predictions[stem], device, config, core_size=core_size
        )
        acceptance_scores[stem] = variants[selected["variant"]]
    selected_edges = {
        stem: link_configuration(
            predictions[stem],
            (
                selection_scores[stem][selected["variant"]]
                if stem in selection_scores
                else acceptance_scores[stem]
            ),
            selected,
        )
        for stem in VALIDATION_STEMS
    }
    acceptance_summary = _score_partition(
        predictions, truths, train_dir, selected_edges, ACCEPTANCE_STEMS
    )
    all_summary = _score_partition(
        predictions, truths, train_dir, selected_edges, VALIDATION_STEMS
    )
    selected = {
        **selected,
        "acceptance_summary": acceptance_summary,
        "all_summary": all_summary,
    }
    selection_delta = (
        selected["selection_summary"]["proxy_score"]
        - base_summary["selection"]["proxy_score"]
    )
    acceptance_delta = (
        acceptance_summary["proxy_score"] - base_summary["acceptance"]["proxy_score"]
    )
    acceptance_passed = bool(
        acceptance_delta > 0
        and acceptance_summary["worst_movie"]
        >= base_summary["acceptance"]["worst_movie"] - 0.01
    )
    payload = {
        "schema_version": 1,
        "model_family": "Higher-Order Cell Tracking Transformer",
        "prediction_graph_kind": "postprocessed_public_comparator_with_transferred_raw_confidence",
        "validation_stems": list(VALIDATION_STEMS),
        "selection_stems": list(SELECTION_STEMS),
        "acceptance_stems": list(ACCEPTANCE_STEMS),
        "selection_rule": "head/method/thresholds selected on selection movies; acceptance scored once after freeze",
        "base_graph": base_summary,
        "selected": selected,
        "selection_delta_vs_base": selection_delta,
        "acceptance_delta_vs_base": acceptance_delta,
        "acceptance_passed": acceptance_passed,
        "stored_edge_probability_evidence": summarize_stored_edge_probabilities(
            predictions
        ),
        "configurations": configurations,
    }
    atomic_json(output_dir / "complete_movie_validation.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--processed-validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-prefix", type=int, default=128)
    parser.add_argument("--core-size", type=float, default=128.0)
    parser.add_argument("--max-wall-seconds", type=float, default=13_200.0)
    args = parser.parse_args()

    started = time.monotonic()
    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for HOCT feature extraction")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = torch.jit.load(str(args.pretrained_model), map_location=device).eval()
    config = ProbeConfig()
    train_dir = args.competition_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Competition train directory not found: {train_dir}")
    excluded = set(VALIDATION_STEMS)
    by_prefix: dict[str, list[Path]] = {}
    for path in train_dir.glob("*.geff"):
        if path.stem not in excluded:
            by_prefix.setdefault(path.stem.split("_")[0], []).append(path)
    train_paths: list[Path] = []
    for prefix, paths in sorted(by_prefix.items()):
        ranked = sorted(
            paths,
            key=lambda path: hashlib.sha256(path.stem.encode()).hexdigest(),
        )
        selected = ranked[: args.train_per_prefix]
        train_paths.extend(selected)
        print(f"HOCT TRAIN SELECT: {prefix} -> {len(selected)} movies", flush=True)
    if len(train_paths) != 195:
        raise RuntimeError(f"Expected all 195 non-validation movies, found {len(train_paths)}")

    features, labels, extraction = extract_training_examples(
        model, train_paths, device, config
    )
    if extraction["candidate_recall"] < 0.98:
        raise RuntimeError(f"HOCT candidate recall is too low: {extraction}")
    probe, fitting = fit_probe(features, labels, model, device, config)
    checkpoint = {
        "schema_version": 1,
        "model_family": "HOCT linear probe",
        "source_model_sha256": sha256_file(args.pretrained_model),
        "source_model_name": args.pretrained_model.name,
        "head_weight": probe.weight.detach().cpu(),
        "head_bias": probe.bias.detach().cpu(),
        "probe_config": asdict(config),
        "extraction": extraction,
        "fitting": fitting,
    }
    checkpoint_path = args.output_dir / "hoct_probe.pt"
    torch.save(checkpoint, checkpoint_path)

    processed_videos = read_submission(args.processed_validation_csv)
    raw_videos = read_raw_graphs(args.validation_predictions, set(processed_videos))
    transfer = transfer_raw_edge_probabilities(processed_videos, raw_videos)
    validation = validate_complete_movies(
        model,
        probe,
        processed_videos,
        train_dir,
        device,
        config,
        args.output_dir,
        np.full(3, args.core_size, dtype=np.float32),
    )
    elapsed = time.monotonic() - started
    if elapsed > args.max_wall_seconds:
        raise TimeoutError(
            f"HOCT run exceeded declared wall budget: {elapsed:.1f}s"
        )
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "source_model_sha256": sha256_file(args.pretrained_model),
        "probe_sha256": sha256_file(checkpoint_path),
        "probe_config": asdict(config),
        "feature_extraction": extraction,
        "probe_fitting": fitting,
        "edge_probability_transfer": transfer,
        "complete_movie_validation_sha256": sha256_file(
            args.output_dir / "complete_movie_validation.json"
        ),
        "selected": validation["selected"],
        "selection_delta_vs_base": validation["selection_delta_vs_base"],
        "acceptance_delta_vs_base": validation["acceptance_delta_vs_base"],
        "association_acceptance_passed": validation["acceptance_passed"],
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
