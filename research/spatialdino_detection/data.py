"""Leakage-safe Biohub training inventory for SpatialDINO detection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


VALIDATION_STEMS = frozenset(
    {
        "44b6_d29c9ab2",
        "44b6_3a861e03",
        "44b6_d5e7d891",
        "44b6_ddf577ad",
        "6bba_09961292",
        "6bba_bb9f20c3",
        "6bba_784a78c9",
        "6bba_57b7cc1e",
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)


@dataclass(frozen=True)
class MovieRecord:
    stem: str
    image_path: Path
    frame_count: int
    q_low: float
    q_high: float
    annotations: dict[int, np.ndarray]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select_frame_pairs(
    frame_counts: dict[str, int], *, pairs_per_movie: int, seed: int
) -> list[tuple[str, int]]:
    if pairs_per_movie <= 0:
        raise ValueError("pairs_per_movie must be positive")
    selected: list[tuple[str, int]] = []
    for stem, frame_count in sorted(frame_counts.items()):
        if stem in VALIDATION_STEMS:
            continue
        if frame_count < 2:
            raise ValueError(f"movie {stem} has fewer than two frames")
        candidates = np.linspace(
            0, frame_count - 2, num=min(pairs_per_movie, frame_count - 1), dtype=int
        )
        selected.extend((stem, int(frame)) for frame in np.unique(candidates))
    if not selected:
        raise RuntimeError("no non-validation frame pairs were selected")
    random.Random(seed).shuffle(selected)
    return selected


def _graph_points_by_frame(path: Path) -> dict[int, np.ndarray]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    graph = graph[0] if isinstance(graph, tuple) else graph
    points: dict[int, list[tuple[float, float, float]]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        points.setdefault(int(row["t"]), []).append(
            (float(row["z"]), float(row["y"]), float(row["x"]))
        )
    return {
        frame: np.asarray(coords, dtype=np.float32).reshape(-1, 3)
        for frame, coords in points.items()
    }


def discover_movies(train_dir: Path) -> list[MovieRecord]:
    from biohub_tracking.io import open_dataset

    records: list[MovieRecord] = []
    for image_path in sorted(train_dir.glob("*.zarr")):
        stem = image_path.stem
        if stem in VALIDATION_STEMS:
            continue
        truth_path = train_dir / f"{stem}.geff"
        if not truth_path.is_dir():
            raise FileNotFoundError(truth_path)
        dataset = open_dataset(
            image_path, normalize=False, load_image=False, require_tracks=False
        )
        if "0.001" not in dataset.quantiles or "0.999" not in dataset.quantiles:
            raise ValueError(f"missing image quantiles for {stem}")
        records.append(
            MovieRecord(
                stem=stem,
                image_path=image_path,
                frame_count=int(dataset.image_shape[0]),
                q_low=float(dataset.quantiles["0.001"]),
                q_high=float(dataset.quantiles["0.999"]),
                annotations=_graph_points_by_frame(truth_path),
            )
        )
    if not records or any(record.stem in VALIDATION_STEMS for record in records):
        raise RuntimeError("training discovery violated the frozen validation split")
    return records
