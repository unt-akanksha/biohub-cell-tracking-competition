from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from synthetic_data import corrected_sequence_graph, division_prior_weight
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.synthetic_pretrain.data import (
        corrected_sequence_graph,
        division_prior_weight,
    )


VOXEL_SCALE_UM = np.array((1.625, 0.40625, 0.40625), dtype=np.float32)
MODEL_SPATIAL_SCALE = VOXEL_SCALE_UM / VOXEL_SCALE_UM[-1]
VALIDATION_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
BASELINE_PROXY = 0.9294432421394134


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GraphVideo:
    stem: str
    node_ids: np.ndarray
    times: np.ndarray
    coords_voxel: np.ndarray
    edges: np.ndarray
    ids_by_time: dict[int, np.ndarray] = field(init=False)
    coord_by_id: dict[int, np.ndarray] = field(init=False)
    time_by_id: dict[int, int] = field(init=False)
    edge_set: set[tuple[int, int]] = field(init=False)
    division_sources: set[int] = field(init=False)
    window_starts: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        self.times = np.asarray(self.times, dtype=np.int32)
        self.coords_voxel = np.asarray(self.coords_voxel, dtype=np.float32)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        if len(self.node_ids) != len(self.times) or len(self.node_ids) != len(self.coords_voxel):
            raise ValueError(f"{self.stem}: inconsistent node arrays")
        if len(np.unique(self.node_ids)) != len(self.node_ids):
            raise ValueError(f"{self.stem}: duplicate node IDs")

        self.ids_by_time = {
            int(t): self.node_ids[self.times == t] for t in np.unique(self.times)
        }
        self.coord_by_id = {
            int(node_id): coord
            for node_id, coord in zip(self.node_ids.tolist(), self.coords_voxel)
        }
        self.time_by_id = {
            int(node_id): int(t) for node_id, t in zip(self.node_ids.tolist(), self.times.tolist())
        }
        self.edge_set = {(int(s), int(t)) for s, t in self.edges.tolist()}
        out_degree: dict[int, int] = {}
        for source, _target in self.edge_set:
            out_degree[source] = out_degree.get(source, 0) + 1
        self.division_sources = {source for source, degree in out_degree.items() if degree >= 2}

        starts: list[int] = []
        if len(self.times):
            edge_times = {
                self.time_by_id[source]
                for source, target in self.edge_set
                if source in self.time_by_id
                and target in self.time_by_id
                and self.time_by_id[target] == self.time_by_id[source] + 1
            }
            for start in range(int(self.times.min()), int(self.times.max()) - 2):
                if any(start <= t < start + 3 for t in edge_times):
                    starts.append(start)
        self.window_starts = tuple(starts)

    @property
    def scaled_coords(self) -> np.ndarray:
        return self.coords_voxel * MODEL_SPATIAL_SCALE[None]


def graph_from_geff(path: Path):
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def read_graph_video(path: Path) -> GraphVideo:
    graph = graph_from_geff(path)
    node_ids: list[int] = []
    times: list[int] = []
    coords: list[tuple[float, float, float]] = []
    for row in graph.node_attrs().iter_rows(named=True):
        node_ids.append(int(row["node_id"]))
        times.append(int(row["t"]))
        coords.append((float(row["z"]), float(row["y"]), float(row["x"])))
    edges = [
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return GraphVideo(
        stem=path.stem,
        node_ids=np.asarray(node_ids),
        times=np.asarray(times),
        coords_voxel=np.asarray(coords),
        edges=np.asarray(edges),
    )


def select_synthetic_sequence_paths(root: Path, limit: int) -> tuple[Path, list[Path]]:
    manifests = []
    for path in ([root / "manifest.json"] if (root / "manifest.json").is_file() else root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("sequences"):
            manifests.append((path, payload))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"Expected one synthetic sequence manifest below {root}, found "
            f"{[str(path) for path, _payload in manifests]}"
        )
    manifest_path, payload = manifests[0]
    records = [
        record
        for record in payload["sequences"]
        if int(record.get("T", 0)) >= 4 and int(record.get("n_edges", 0)) > 0
    ]
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(str(record["file"]).encode("utf-8")).hexdigest(),
    )
    selected = [manifest_path.parent / record["file"] for record in ranked[:limit]]
    if not selected or any(not path.is_file() for path in selected):
        raise FileNotFoundError("synthetic sequence selection contains missing files")
    return manifest_path, selected


def read_synthetic_graph_video(path: Path) -> GraphVideo:
    graph = corrected_sequence_graph(path)
    node_ids = np.arange(len(graph.nodes), dtype=np.int64)
    return GraphVideo(
        stem=f"synthetic_{path.stem}",
        node_ids=node_ids,
        times=graph.nodes[:, 0].astype(np.int32),
        coords_voxel=graph.nodes[:, 1:4],
        edges=graph.edges,
    )


def select_training_paths(
    train_dir: Path,
    validation_stems: Iterable[str],
    per_prefix: int,
) -> list[Path]:
    excluded = set(validation_stems)
    by_prefix: dict[str, list[Path]] = {}
    for path in train_dir.glob("*.geff"):
        if path.stem in excluded:
            continue
        by_prefix.setdefault(path.stem.split("_")[0], []).append(path)
    selected: list[Path] = []
    for prefix, paths in sorted(by_prefix.items()):
        ranked = sorted(
            paths,
            key=lambda p: hashlib.sha256(p.stem.encode("utf-8")).hexdigest(),
        )
        selected.extend(ranked[:per_prefix])
        print(f"TRAIN SELECT: {prefix} -> {min(per_prefix, len(ranked))} movies", flush=True)
    return selected


@dataclass
class WindowSample:
    coords: np.ndarray
    features: np.ndarray
    target: np.ndarray
    valid_mask: np.ndarray
    division_target: np.ndarray
    positive_edges: int


def point_features(coords: np.ndarray, timepoints: np.ndarray) -> np.ndarray:
    """Build 12 Biohub-native shallow features with Trackastra-compatible scale."""
    features = np.zeros((len(coords), 12), dtype=np.float32)
    for timepoint in np.unique(timepoints):
        indices = np.flatnonzero(timepoints == timepoint)
        points = coords[indices, 1:]
        n = len(points)
        if n == 1:
            features[indices, 0] = 5.0
            features[indices, 1] = 0.0
            features[indices, 2:11] = np.eye(3, dtype=np.float32).reshape(1, 9)
            continue

        tree = cKDTree(points)
        k = min(9, n)
        distances, neighbors = tree.query(points, k=k)
        if k == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        nearest = distances[:, 1:] if k > 1 else np.full((n, 1), 20.0)
        mean_nearest = np.maximum(nearest.mean(axis=1), 1e-3)
        features[indices, 0] = np.clip(0.5 * mean_nearest, 2.0, 20.0)
        features[indices, 1] = np.exp(-mean_nearest / 20.0)
        for local_index, global_index in enumerate(indices):
            neighbor_points = points[np.atleast_1d(neighbors[local_index])[1:]]
            if len(neighbor_points) < 2:
                covariance = np.eye(3, dtype=np.float32)
            else:
                delta = neighbor_points - points[local_index]
                covariance = (delta.T @ delta) / max(len(delta), 1)
            features[global_index, 2:11] = np.clip(covariance.reshape(-1), -1000, 1000)

    # The tile coordinate system has no biologically meaningful image border.
    features[:, 11] = 0.0
    return features


def _sample_once(
    video: GraphVideo,
    rng: np.random.Generator,
    *,
    window: int,
    max_tokens: int,
    tile_radius: np.ndarray,
    drop_probability: float,
    false_positive_probability: float,
    jitter_sigma: float,
    prefer_division_probability: float,
    hard_negative_radius: float,
) -> WindowSample | None:
    if not video.window_starts:
        return None
    start = int(rng.choice(video.window_starts))
    end = start + window
    selected_ids = np.concatenate(
        [video.ids_by_time.get(t, np.empty(0, dtype=np.int64)) for t in range(start, end)]
    )
    if len(selected_ids) < 2:
        return None

    division_candidates = [
        source
        for source in video.division_sources
        if start <= video.time_by_id.get(source, -1000) < end - 1
        and source in set(selected_ids.tolist())
    ]
    if division_candidates and rng.random() < prefer_division_probability:
        anchor_id = int(rng.choice(division_candidates))
    else:
        anchor_id = int(rng.choice(selected_ids))
    anchor = video.coord_by_id[anchor_id] * MODEL_SPATIAL_SCALE

    scaled = np.stack([video.coord_by_id[int(node_id)] for node_id in selected_ids])
    scaled = scaled * MODEL_SPATIAL_SCALE[None]
    tile_mask = np.all(np.abs(scaled - anchor[None]) <= tile_radius[None], axis=1)
    tile_ids = selected_ids[tile_mask]
    tile_scaled = scaled[tile_mask]
    if len(tile_ids) < 8:
        tile_ids = selected_ids
        tile_scaled = scaled
    if len(tile_ids) > max_tokens:
        distances = np.linalg.norm((tile_scaled - anchor[None]) / tile_radius[None], axis=1)
        keep = np.argsort(distances, kind="stable")[:max_tokens]
        tile_ids = tile_ids[keep]
        tile_scaled = tile_scaled[keep]

    origin_ids = tile_ids.astype(np.int64, copy=True)
    node_times = np.array([video.time_by_id[int(node_id)] for node_id in origin_ids], dtype=np.int32)
    if drop_probability > 0:
        keep = rng.random(len(origin_ids)) >= drop_probability
        if keep.sum() < min(8, len(origin_ids)):
            keep[np.argsort(rng.random(len(origin_ids)))[: min(8, len(origin_ids))]] = True
        origin_ids = origin_ids[keep]
        node_times = node_times[keep]
        tile_scaled = tile_scaled[keep]

    if jitter_sigma > 0:
        tile_scaled = tile_scaled + rng.normal(0.0, jitter_sigma, tile_scaled.shape).astype(np.float32)

    n_false = int(rng.binomial(len(origin_ids), false_positive_probability))
    if n_false and len(origin_ids) + n_false <= max_tokens:
        parents = rng.integers(0, len(origin_ids), size=n_false)
        false_coords = tile_scaled[parents] + rng.normal(0.0, 8.0, (n_false, 3)).astype(np.float32)
        false_times = node_times[parents]
        tile_scaled = np.concatenate([tile_scaled, false_coords], axis=0)
        node_times = np.concatenate([node_times, false_times], axis=0)
        origin_ids = np.concatenate([origin_ids, np.full(n_false, -1, dtype=np.int64)])

    order = np.lexsort((tile_scaled[:, 2], tile_scaled[:, 1], tile_scaled[:, 0], node_times))
    tile_scaled = tile_scaled[order]
    node_times = node_times[order]
    origin_ids = origin_ids[order]
    spatial_center = np.median(tile_scaled, axis=0)
    coords = np.concatenate(
        [
            (node_times - start).astype(np.float32)[:, None],
            (tile_scaled - spatial_center[None]).astype(np.float32),
        ],
        axis=1,
    )

    n = len(coords)
    target = np.zeros((n, n), dtype=np.float32)
    division_target = np.zeros((n, n), dtype=np.float32)
    index_by_origin = {
        int(origin_id): index
        for index, origin_id in enumerate(origin_ids.tolist())
        if origin_id >= 0
    }
    for source_id, target_id in video.edge_set:
        source_index = index_by_origin.get(source_id)
        target_index = index_by_origin.get(target_id)
        if source_index is None or target_index is None:
            continue
        target[source_index, target_index] = 1.0
        if source_id in video.division_sources:
            division_target[source_index, target_index] = 1.0

    dt = coords[:, None, 0] - coords[None, :, 0]
    # Rows are sources and columns are targets: target time - source time == 1.
    forward = dt == -1
    distances = np.linalg.norm(coords[:, None, 1:] - coords[None, :, 1:], axis=-1)
    valid = forward & (distances <= hard_negative_radius)
    valid |= target.astype(bool)
    positives = int(target.sum())
    if positives == 0 or not valid.any():
        return None
    features = point_features(coords, coords[:, 0].astype(np.int32))
    return WindowSample(coords, features, target, valid, division_target, positives)


def sample_window(
    videos: list[GraphVideo],
    rng: np.random.Generator,
    **kwargs: Any,
) -> WindowSample:
    for _attempt in range(40):
        video = videos[int(rng.integers(0, len(videos)))]
        sample = _sample_once(video, rng, **kwargs)
        if sample is not None:
            return sample
    raise RuntimeError("Could not draw a nonempty positive training window")


def association_loss(
    model,
    sample: WindowSample,
    device: torch.device,
    *,
    division_weight_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    coords = torch.from_numpy(sample.coords).unsqueeze(0).to(device)
    features = torch.from_numpy(sample.features).unsqueeze(0).to(device)
    target = torch.from_numpy(sample.target).to(device)
    valid = torch.from_numpy(sample.valid_mask).to(device)
    division = torch.from_numpy(sample.division_target).to(device)

    logits = model(coords, features)[0]
    timepoints = coords[:, :, 0].long()
    probabilities = model.normalize_output(logits.float().unsqueeze(0), timepoints, coords.float())[0]
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(probabilities, target, reduction="none")
    focal = torch.where(target > 0, (1.0 - probabilities).square(), probabilities.square())
    weights = 1.0 + 4.0 * target + 8.0 * division * division_weight_scale
    loss = (bce * focal * weights)[valid].mean()

    with torch.no_grad():
        positive_probability = float(probabilities[target > 0].mean().item())
        negative_probability = float(probabilities[valid & (target == 0)].mean().item())
    return loss, {
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "positive_edges": float(sample.positive_edges),
    }


@torch.no_grad()
def evaluate_sample_loss(
    model,
    videos: list[GraphVideo],
    device: torch.device,
    seed: int,
    n_samples: int,
    sample_kwargs: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    values: list[dict[str, float]] = []
    losses: list[float] = []
    for _ in range(n_samples):
        sample = sample_window(
            videos,
            rng,
            **{
                **sample_kwargs,
                "drop_probability": 0.0,
                "false_positive_probability": 0.0,
                "jitter_sigma": 0.0,
                "prefer_division_probability": 0.5,
            },
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, stats = association_loss(model, sample, device)
        losses.append(float(loss.item()))
        values.append(stats)
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "positive_probability": float(np.mean([v["positive_probability"] for v in values])),
        "negative_probability": float(np.mean([v["negative_probability"] for v in values])),
        "samples": n_samples,
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(_plain(row))


def _recursive_source_tiles(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_tokens: int,
    radius: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    target_tree = cKDTree(target_points) if len(target_points) else None
    pending = [np.arange(len(source_points), dtype=np.int64)]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    while pending:
        source_indices = pending.pop()
        if target_tree is None:
            target_indices = np.empty(0, dtype=np.int64)
        else:
            neighborhoods = target_tree.query_ball_point(source_points[source_indices], r=radius)
            target_indices = np.array(
                sorted({index for neighborhood in neighborhoods for index in neighborhood}),
                dtype=np.int64,
            )
        if len(source_indices) + len(target_indices) <= max_tokens:
            result.append((source_indices, target_indices))
            continue
        if len(source_indices) <= 1:
            distances = np.linalg.norm(target_points - source_points[source_indices[0]], axis=1)
            target_indices = np.argsort(distances)[: max(0, max_tokens - 1)]
            result.append((source_indices, target_indices.astype(np.int64)))
            continue
        span = np.ptp(source_points[source_indices], axis=0)
        axis = int(np.argmax(span))
        ordered = source_indices[np.argsort(source_points[source_indices, axis], kind="stable")]
        midpoint = max(1, len(ordered) // 2)
        pending.append(ordered[midpoint:])
        pending.append(ordered[:midpoint])
    return result


@torch.no_grad()
def predict_pair_scores(
    model,
    source_coords_voxel: np.ndarray,
    target_coords_voxel: np.ndarray,
    device: torch.device,
    *,
    max_tokens: int,
    candidate_radius: float,
) -> np.ndarray:
    source_points = source_coords_voxel * MODEL_SPATIAL_SCALE[None]
    target_points = target_coords_voxel * MODEL_SPATIAL_SCALE[None]
    scores = np.full((len(source_points), len(target_points)), -np.inf, dtype=np.float32)
    for source_indices, target_indices in _recursive_source_tiles(
        source_points, target_points, max_tokens, candidate_radius
    ):
        if not len(source_indices) or not len(target_indices):
            continue
        source = source_points[source_indices]
        target = target_points[target_indices]
        center = np.median(np.concatenate([source, target], axis=0), axis=0)
        coords_np = np.concatenate(
            [
                np.concatenate([np.zeros((len(source), 1), dtype=np.float32), source - center], axis=1),
                np.concatenate([np.ones((len(target), 1), dtype=np.float32), target - center], axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        features_np = point_features(coords_np, coords_np[:, 0].astype(np.int32))
        coords = torch.from_numpy(coords_np).unsqueeze(0).to(device)
        features = torch.from_numpy(features_np).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(coords, features)
        probs = model.normalize_output(
            logits.float(), coords[:, :, 0].long(), coords.float()
        )[0]
        block = probs[: len(source), len(source) :].detach().cpu().numpy().astype(np.float32)
        scores[np.ix_(source_indices, target_indices)] = np.maximum(
            scores[np.ix_(source_indices, target_indices)], block
        )
    return scores


@torch.no_grad()
def predict_movie_scores(
    model,
    video: GraphVideo,
    device: torch.device,
    *,
    max_tokens: int,
    candidate_radius: float,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model.eval()
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for t in sorted(video.ids_by_time):
        source_ids = video.ids_by_time[t]
        target_ids = video.ids_by_time.get(t + 1)
        if target_ids is None or not len(source_ids) or not len(target_ids):
            continue
        source_coords = np.stack([video.coord_by_id[int(i)] for i in source_ids])
        target_coords = np.stack([video.coord_by_id[int(i)] for i in target_ids])
        score = predict_pair_scores(
            model,
            source_coords,
            target_coords,
            device,
            max_tokens=max_tokens,
            candidate_radius=candidate_radius,
        )
        result[t] = (source_ids, target_ids, score)
    return result


def link_movie(
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    edge_threshold: float,
    division_threshold: float,
    division_ratio: float,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for _t, (source_ids, target_ids, raw_scores) in sorted(pair_scores.items()):
        scores = np.where(np.isfinite(raw_scores), raw_scores, -1e6)
        rows, cols = linear_sum_assignment(-scores)
        used_targets: set[int] = set()
        primary_by_source: dict[int, tuple[int, float]] = {}
        for row, col in zip(rows.tolist(), cols.tolist()):
            score = float(scores[row, col])
            if score < edge_threshold:
                continue
            source_id = int(source_ids[row])
            target_id = int(target_ids[col])
            edges.append((source_id, target_id))
            used_targets.add(col)
            primary_by_source[row] = (col, score)

        division_candidates: list[tuple[float, int, int]] = []
        for row, (_primary_col, primary_score) in primary_by_source.items():
            ranked = np.argsort(scores[row])[::-1]
            for col in ranked.tolist():
                if col in used_targets or col == _primary_col:
                    continue
                score = float(scores[row, col])
                if score < division_threshold or score < primary_score * division_ratio:
                    break
                division_candidates.append((score, row, col))
                break
        for score, row, col in sorted(division_candidates, reverse=True):
            if col in used_targets:
                continue
            edges.append((int(source_ids[row]), int(target_ids[col])))
            used_targets.add(col)
    return edges


def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_rest) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_rest) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, pred_ids in pred_by_t.items():
        gt_ids = gt_by_t.get(t, [])
        if not gt_ids:
            continue
        pred_pos = np.array([pred_nodes[p][1:] for p in pred_ids]) * VOXEL_SCALE_UM
        gt_pos = np.array([gt_nodes[g][1:] for g in gt_ids]) * VOXEL_SCALE_UM
        cost = np.linalg.norm(pred_pos[:, None] - gt_pos[None], axis=-1)
        gated = np.where(cost <= max_dist, cost, 1e6)
        rows, cols = linear_sum_assignment(gated)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if gated[row, col] >= 1e6:
                continue
            pred_to_gt[pred_ids[row]] = gt_ids[col]
            gt_to_pred[gt_ids[col]] = pred_ids[row]
    return pred_to_gt, gt_to_pred


def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for source, target in gt_edge_set:
        gt_outgoing.setdefault(source, set()).add(target)
        gt_incoming_source[target] = source
    tp = fp = 0
    matched: set[tuple[int, int]] = set()
    for source, target in pred_edges:
        matched_source = pred_to_gt.get(source)
        matched_target = pred_to_gt.get(target)
        if matched_source is not None and matched_target in gt_outgoing.get(matched_source, ()):
            tp += 1
            matched.add((matched_source, matched_target))
        elif (
            matched_target is not None and matched_target in gt_incoming_source
        ) or (matched_source is not None and bool(gt_outgoing.get(matched_source))):
            fp += 1
    return tp, fp, len(gt_edge_set - matched)


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true: int | float | None, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))


def weakly_connected_components(node_ids, edges):
    parent = {node: node for node in node_ids}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for source, target in edges:
        if source in parent and target in parent:
            union(source, target)
    return {node: find(node) for node in node_ids}


def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for source, target in gt_edges:
        gt_out.setdefault(source, set()).add(target)
        gt_in[target] = source
    pred_out: dict[int, set[int]] = {}
    for source, target in pred_edges:
        pred_out.setdefault(source, set()).add(target)
    components = weakly_connected_components(list(pred_nodes), list(pred_edges))
    fork_components = {
        components[node]
        for node, targets in pred_out.items()
        if len(targets) >= 2 and node in components
    }

    def descendants(root: int) -> set[int]:
        seen = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for nxt in gt_out.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = fn = 0
    matched_sources: set[int] = set()
    for gt_source, targets in gt_out.items():
        if len(targets) < 2:
            continue
        children = sorted(targets)[:2]
        anchors = [gt_source] + ([gt_in[gt_source]] if gt_source in gt_in else [])
        anchor_pred = [gt_to_pred[a] for a in anchors if a in gt_to_pred]
        lineage_components = []
        for child in children:
            hit = {
                components[pred_id]
                for gt_id in descendants(child)
                if (pred_id := gt_to_pred.get(gt_id)) is not None and pred_id in components
            }
            lineage_components.append(hit)
        anchor_components = {components[p] for p in anchor_pred if p in components}
        found = bool(lineage_components[0] and lineage_components[1]) and any(
            component in lineage_components[0]
            and component in lineage_components[1]
            and component in fork_components
            for component in anchor_components
        )
        if found:
            tp += 1
            matched_sources.add(gt_source)
        else:
            fn += 1
    fp = 0
    for node, targets in pred_out.items():
        if len(targets) < 2:
            continue
        gt_node = pred_to_gt.get(node)
        if gt_node is not None and gt_node in gt_out and gt_node not in matched_sources:
            fp += 1
    return tp, fp, fn


def _find_key_recursive(value: Any, key: str):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key_recursive(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key_recursive(child, key)
            if found is not None:
                return found
    return None


def read_true_node_count(geff_path: Path) -> float | None:
    for json_path in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not json_path.exists():
            continue
        try:
            value = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("estimated_number_of_nodes", "estimated_num_nodes", "number_of_nodes"):
            found = _find_key_recursive(value, key)
            if found is not None:
                try:
                    return float(found)
                except (TypeError, ValueError):
                    pass
    return None


def video_plain(video: GraphVideo) -> tuple[dict[int, tuple], list[tuple[int, int]]]:
    nodes = {
        int(node_id): (int(t), float(coord[0]), float(coord[1]), float(coord[2]))
        for node_id, t, coord in zip(video.node_ids, video.times, video.coords_voxel)
    }
    return nodes, [(int(s), int(t)) for s, t in video.edges.tolist()]


def score_linked_video(pred: GraphVideo, gt: GraphVideo, pred_edges, t_true) -> dict[str, Any]:
    pred_nodes, _raw_edges = video_plain(pred)
    gt_nodes, gt_edges = video_plain(gt)
    pred_to_gt, gt_to_pred = match_nodes_bipartite(pred_nodes, gt_nodes)
    tp, fp, fn = compute_edge_confusion(pred_edges, gt_edges, pred_to_gt)
    raw_jaccard = edge_jaccard(tp, fp, fn)
    adjusted = adjusted_jaccard(raw_jaccard, len(pred_nodes), t_true)
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
    )
    return {
        "stem": pred.stem,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_jaccard": raw_jaccard,
        "adjusted_edge_jaccard": adjusted,
        "t_pred": len(pred_nodes),
        "t_true": t_true,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
        "division_jaccard": edge_jaccard(div_tp, div_fp, div_fn),
        "weight": tp + fp + fn,
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_weight = sum(row["weight"] for row in rows) or 1
    adjusted = sum(row["adjusted_edge_jaccard"] * row["weight"] for row in rows) / total_weight
    div_tp = sum(row["div_tp"] for row in rows)
    div_fp = sum(row["div_fp"] for row in rows)
    div_fn = sum(row["div_fn"] for row in rows)
    division = edge_jaccard(div_tp, div_fp, div_fn)
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + 0.1 * division,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
        "n_samples": len(rows),
        "worst_movie": min((row["adjusted_edge_jaccard"] for row in rows), default=0.0),
    }


def complete_movie_validation(
    model,
    validation_predictions: Path,
    train_dir: Path,
    device: torch.device,
    output_dir: Path,
    max_tokens: int,
    candidate_radius: float,
) -> dict[str, Any]:
    prediction_paths = {
        path.stem: path for path in validation_predictions.rglob("*.geff")
    }
    predictions: dict[str, GraphVideo] = {}
    truths: dict[str, GraphVideo] = {}
    pair_scores: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for stem in VALIDATION_STEMS:
        if stem not in prediction_paths:
            raise FileNotFoundError(f"Missing validator prediction graph for {stem}")
        predictions[stem] = read_graph_video(prediction_paths[stem])
        truths[stem] = read_graph_video(train_dir / f"{stem}.geff")
        print(f"COMPLETE MOVIE INFERENCE: {stem}", flush=True)
        pair_scores[stem] = predict_movie_scores(
            model,
            predictions[stem],
            device,
            max_tokens=max_tokens,
            candidate_radius=candidate_radius,
        )

    configurations: list[dict[str, Any]] = []
    for edge_threshold in (0.03, 0.05, 0.08, 0.12, 0.18):
        for division_threshold in (0.05, 0.08, 0.12, 0.18, 0.25):
            for division_ratio in (0.25, 0.50):
                rows: list[dict[str, Any]] = []
                for stem in VALIDATION_STEMS:
                    edges = link_movie(
                        pair_scores[stem],
                        edge_threshold=edge_threshold,
                        division_threshold=division_threshold,
                        division_ratio=division_ratio,
                    )
                    rows.append(
                        score_linked_video(
                            predictions[stem],
                            truths[stem],
                            edges,
                            read_true_node_count(train_dir / f"{stem}.geff"),
                        )
                    )
                summary = aggregate_scores(rows)
                configurations.append(
                    {
                        "edge_threshold": edge_threshold,
                        "division_threshold": division_threshold,
                        "division_ratio": division_ratio,
                        "summary": summary,
                        "movies": rows,
                    }
                )
    best = max(
        configurations,
        key=lambda item: (
            item["summary"]["proxy_score"],
            item["summary"]["worst_movie"],
            -item["summary"]["div_fp"],
        ),
    )
    payload = {
        "schema_version": 1,
        "validation_stems": list(VALIDATION_STEMS),
        "selection": "complete-movie official-formula proxy; public leaderboard unused",
        "baseline_proxy": BASELINE_PROXY,
        "best": best,
        "delta_proxy_vs_public_0927_baseline": best["summary"]["proxy_score"] - BASELINE_PROXY,
        "configurations": configurations,
    }
    atomic_json(output_dir / "complete_movie_validation.json", payload)
    return payload


def build_synthetic_video() -> GraphVideo:
    node_ids: list[int] = []
    times: list[int] = []
    coords: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    next_id = 1
    previous: list[int] = []
    for t in range(8):
        current = []
        for cell in range(6):
            node_ids.append(next_id)
            times.append(t)
            coords.append((4.0 + cell, 20.0 + cell * 3 + t, 30.0 + cell * 2))
            current.append(next_id)
            next_id += 1
        if previous:
            edges.extend(zip(previous, current))
        previous = current
    # Add one true division into an otherwise linear lineage.
    division_source = node_ids[2 * 6]
    division_target = node_ids[3 * 6 + 1]
    edges.append((division_source, division_target))
    return GraphVideo(
        "synthetic",
        np.asarray(node_ids),
        np.asarray(times),
        np.asarray(coords),
        np.asarray(edges),
    )


def self_test() -> None:
    video = build_synthetic_video()
    sample = sample_window(
        [video],
        np.random.default_rng(17),
        window=4,
        max_tokens=128,
        tile_radius=np.array((96.0, 192.0, 192.0)),
        drop_probability=0.0,
        false_positive_probability=0.0,
        jitter_sigma=0.0,
        prefer_division_probability=1.0,
        hard_negative_radius=64.0,
    )
    assert sample.coords.shape[1] == 4
    assert sample.features.shape == (len(sample.coords), 12)
    assert sample.positive_edges > 0
    assert np.all(sample.target.astype(bool) <= sample.valid_mask)

    perfect_nodes, perfect_edges = video_plain(video)
    pred_to_gt, gt_to_pred = match_nodes_bipartite(perfect_nodes, perfect_nodes)
    tp, fp, fn = compute_edge_confusion(perfect_edges, perfect_edges, pred_to_gt)
    assert (tp, fp, fn) == (len(perfect_edges), 0, 0)
    div = compute_division_confusion(
        perfect_nodes, perfect_edges, perfect_nodes, perfect_edges, pred_to_gt, gt_to_pred
    )
    assert div == (1, 0, 0), div

    source = np.array([[0, 0, 0], [0, 100, 100], [0, 200, 200]], dtype=np.float32)
    target = source + 1
    tiles = _recursive_source_tiles(source, target, max_tokens=4, radius=20)
    assert sum(len(tile[0]) for tile in tiles) == len(source)
    print("SELF_TEST_OK")


def train_main(args: argparse.Namespace) -> None:
    start_time = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trackastra_dir = args.trackastra_dir.resolve()
    sys.path.insert(0, str(trackastra_dir))
    from trackastra.model.model import TrackingTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required for the paid fine-tuning run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.competition_dir / "train"
    training_paths = select_training_paths(
        train_dir, VALIDATION_STEMS, args.train_per_prefix
    )
    validation_paths = [train_dir / f"{stem}.geff" for stem in VALIDATION_STEMS]
    print(f"Loading {len(training_paths)} training graphs...", flush=True)
    training_videos = [read_graph_video(path) for path in training_paths]
    print(f"Loading {len(validation_paths)} clean validation graphs...", flush=True)
    validation_videos = [read_graph_video(path) for path in validation_paths]
    synthetic_manifest: Path | None = None
    synthetic_paths: list[Path] = []
    synthetic_videos: list[GraphVideo] = []
    if args.synthetic_steps > 0:
        if args.synthetic_root is None:
            raise ValueError("--synthetic-root is required when --synthetic-steps is positive")
        if args.resume:
            raise ValueError("resume is disabled for the two-stage synthetic/real schedule")
        synthetic_manifest, synthetic_paths = select_synthetic_sequence_paths(
            args.synthetic_root, args.synthetic_graphs
        )
        print(f"Loading {len(synthetic_paths)} synthetic graph-only sequences...", flush=True)
        synthetic_videos = [read_synthetic_graph_video(path) for path in synthetic_paths]

    model = TrackingTransformer.from_folder(args.pretrained_dir, map_location="cpu").to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count < 20_000_000:
        raise RuntimeError(f"Unexpected Trackastra parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = args.output_dir / "checkpoint_last.pt"
    step = 0
    if checkpoint_path.exists() and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        print(f"Resumed at step {step}", flush=True)

    config = {
        "schema_version": 1,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "pretrained_model_sha256": sha256_file(args.pretrained_dir / "model.pt"),
        "trackastra_source": str(trackastra_dir),
        "training_stems": [video.stem for video in training_videos],
        "validation_stems": list(VALIDATION_STEMS),
        "steps_target": args.steps,
        "max_tokens": args.max_tokens,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "drop_probability": args.drop_probability,
        "false_positive_probability": args.false_positive_probability,
        "jitter_sigma": args.jitter_sigma,
        "hard_negative_radius": args.hard_negative_radius,
        "candidate_radius": args.candidate_radius,
        "synthetic": {
            "enabled": bool(synthetic_videos),
            "manifest": str(synthetic_manifest) if synthetic_manifest else None,
            "manifest_sha256": sha256_file(synthetic_manifest) if synthetic_manifest else None,
            "graphs": len(synthetic_videos),
            "graph_names_sha256": (
                hashlib.sha256(
                    "\n".join(path.name for path in synthetic_paths).encode("utf-8")
                ).hexdigest()
                if synthetic_paths
                else None
            ),
            "steps_target": args.synthetic_steps,
            "learning_rate_multiplier": args.synthetic_learning_rate_multiplier,
            "division_weight_scale": division_prior_weight(),
            "prefer_division_probability": args.synthetic_prefer_division_probability,
            "image_members_loaded": False,
        },
    }
    atomic_json(args.output_dir / "training_config.json", config)
    history_path = args.output_dir / "history.csv"
    rng = np.random.default_rng(args.seed + step)
    sample_kwargs = {
        "window": 4,
        "max_tokens": args.max_tokens,
        "tile_radius": np.array((96.0, 192.0, 192.0), dtype=np.float32),
        "drop_probability": args.drop_probability,
        "false_positive_probability": args.false_positive_probability,
        "jitter_sigma": args.jitter_sigma,
        "prefer_division_probability": args.prefer_division_probability,
        "hard_negative_radius": args.hard_negative_radius,
    }

    initial_validation = evaluate_sample_loss(
        model,
        validation_videos,
        device,
        args.seed + 101,
        args.validation_samples,
        sample_kwargs,
    )
    atomic_json(args.output_dir / "initial_window_validation.json", initial_validation)
    print(f"INITIAL WINDOW VALIDATION: {initial_validation}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop_for_time = False
    synthetic_step = 0
    post_synthetic_validation = None
    if synthetic_videos:
        synthetic_history_path = args.output_dir / "synthetic_history.csv"
        synthetic_rng = np.random.default_rng(args.seed + 7001)
        synthetic_sample_kwargs = {
            **sample_kwargs,
            "prefer_division_probability": args.synthetic_prefer_division_probability,
        }
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * args.synthetic_learning_rate_multiplier
        synthetic_losses: list[float] = []
        while synthetic_step < args.synthetic_steps:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.max_wall_seconds - args.validation_reserve_seconds:
                stop_for_time = True
                print("Stopping synthetic stage to preserve complete-movie validation time.", flush=True)
                break
            sample = sample_window(synthetic_videos, synthetic_rng, **synthetic_sample_kwargs)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                loss, stats = association_loss(
                    model,
                    sample,
                    device,
                    division_weight_scale=division_prior_weight(),
                )
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            synthetic_step += 1
            if synthetic_step % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            synthetic_losses.append(float(loss.item()))
            if synthetic_step % args.log_every == 0:
                row = {
                    "phase": "synthetic",
                    "step": synthetic_step,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                    "train_loss": float(np.mean(synthetic_losses[-args.log_every :])),
                    **stats,
                }
                append_history(synthetic_history_path, row)
                print(f"SYNTHETIC TRAIN {row}", flush=True)

        if synthetic_step % args.gradient_accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        synthetic_model = args.output_dir / "model_synthetic.pt"
        temporary_synthetic = synthetic_model.with_suffix(".tmp")
        torch.save(model.state_dict(), temporary_synthetic)
        temporary_synthetic.replace(synthetic_model)
        post_synthetic_validation = evaluate_sample_loss(
            model,
            validation_videos,
            device,
            args.seed + 202,
            args.validation_samples,
            sample_kwargs,
        )
        atomic_json(
            args.output_dir / "post_synthetic_window_validation.json",
            post_synthetic_validation,
        )
        print(f"POST-SYNTHETIC WINDOW VALIDATION: {post_synthetic_validation}", flush=True)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)

    rolling_loss: list[float] = []
    last_log = time.monotonic()
    while step < args.steps and not stop_for_time:
        elapsed = time.monotonic() - start_time
        if elapsed >= args.max_wall_seconds - args.validation_reserve_seconds:
            stop_for_time = True
            print("Stopping training to preserve complete-movie validation time.", flush=True)
            break
        sample = sample_window(training_videos, rng, **sample_kwargs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, stats = association_loss(model, sample, device)
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        step += 1
        if step % args.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        rolling_loss.append(float(loss.item()))

        if step % args.log_every == 0:
            now = time.monotonic()
            row = {
                "phase": "real",
                "step": step,
                "elapsed_seconds": round(now - start_time, 3),
                "train_loss": float(np.mean(rolling_loss[-args.log_every :])),
                **stats,
            }
            append_history(history_path, row)
            print(f"TRAIN {row}", flush=True)
            last_log = now
        if step % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scaler, step, config)

    if step % args.gradient_accumulation:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    save_checkpoint(checkpoint_path, model, optimizer, scaler, step, config)

    model.eval()
    final_model_path = args.output_dir / "model.pt"
    temporary_model = final_model_path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_model)
    temporary_model.replace(final_model_path)
    shutil.copy2(args.pretrained_dir / "config.yaml", args.output_dir / "config.yaml")
    final_validation = evaluate_sample_loss(
        model,
        validation_videos,
        device,
        args.seed + 303,
        args.validation_samples,
        sample_kwargs,
    )
    atomic_json(args.output_dir / "final_window_validation.json", final_validation)
    print(f"FINAL WINDOW VALIDATION: {final_validation}", flush=True)

    complete = complete_movie_validation(
        model,
        args.validation_predictions,
        train_dir,
        device,
        args.output_dir,
        args.max_tokens,
        args.candidate_radius,
    )
    terminal = {
        "status": "completed",
        "stopped_for_time": stop_for_time,
        "step": step,
        "synthetic_step": synthetic_step,
        "elapsed_seconds": time.monotonic() - start_time,
        "parameter_count": parameter_count,
        "model_sha256": sha256_file(final_model_path),
        "initial_window_validation": initial_validation,
        "post_synthetic_window_validation": post_synthetic_validation,
        "final_window_validation": final_validation,
        "complete_movie_best": complete["best"],
        "delta_proxy_vs_public_0927_baseline": complete[
            "delta_proxy_vs_public_0927_baseline"
        ],
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(_plain(terminal), indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--competition-dir", type=Path)
    result.add_argument("--trackastra-dir", type=Path)
    result.add_argument("--pretrained-dir", type=Path)
    result.add_argument("--validation-predictions", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/trackastra_graph_v1"))
    result.add_argument("--seed", type=int, default=1701)
    result.add_argument("--steps", type=int, default=6000)
    result.add_argument("--train-per-prefix", type=int, default=24)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--gradient-accumulation", type=int, default=4)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--drop-probability", type=float, default=0.04)
    result.add_argument("--false-positive-probability", type=float, default=0.03)
    result.add_argument("--jitter-sigma", type=float, default=2.0)
    result.add_argument("--prefer-division-probability", type=float, default=0.5)
    result.add_argument("--hard-negative-radius", type=float, default=64.0)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--synthetic-root", type=Path)
    result.add_argument("--synthetic-graphs", type=int, default=384)
    result.add_argument("--synthetic-steps", type=int, default=0)
    result.add_argument("--synthetic-learning-rate-multiplier", type=float, default=2.0)
    result.add_argument("--synthetic-prefer-division-probability", type=float, default=0.15)
    result.add_argument("--checkpoint-every", type=int, default=250)
    result.add_argument("--log-every", type=int, default=25)
    result.add_argument("--validation-samples", type=int, default=24)
    result.add_argument("--max-wall-seconds", type=int, default=12600)
    result.add_argument("--validation-reserve-seconds", type=int, default=1800)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--allow-cpu", action="store_true")
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        "competition_dir",
        "trackastra_dir",
        "pretrained_dir",
        "validation_predictions",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {missing}")
    train_main(args)


if __name__ == "__main__":
    main()
