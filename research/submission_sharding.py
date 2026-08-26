"""Mandatory two-GPU movie sharding for future Biohub submission kernels."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MovieShard:
    shard_index: int
    shard_count: int
    cuda_token: str
    movie_ids: tuple[str, ...]


def visible_cuda_tokens(
    *,
    detected_devices: int,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve parent-visible CUDA tokens without remapping GPU UUIDs."""

    if detected_devices < 0:
        raise ValueError("detected_devices must be nonnegative")
    environment = os.environ if environ is None else environ
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return tuple(str(index) for index in range(detected_devices))
    tokens = tuple(token.strip() for token in visible.split(",") if token.strip())
    if any(token == "-1" for token in tokens):
        return ()
    if len(tokens) < detected_devices:
        raise ValueError("CUDA visibility exposes fewer tokens than detected devices")
    return tokens[:detected_devices]


def build_movie_shards(
    movie_ids: Sequence[str],
    cuda_tokens: Sequence[str],
    *,
    required_devices: int = 2,
) -> tuple[MovieShard, ...]:
    """Assign each movie exactly once across the required GPU workers."""

    if required_devices != 2:
        raise ValueError("Biohub submission policy requires exactly two devices")
    movies = tuple(str(movie).strip() for movie in movie_ids)
    tokens = tuple(str(token).strip() for token in cuda_tokens)
    if len(tokens) < required_devices:
        raise RuntimeError(
            f"submission requires {required_devices} CUDA devices, found {len(tokens)}"
        )
    if not movies or any(not movie for movie in movies):
        raise ValueError("movie identifiers must be nonempty")
    if len(set(movies)) != len(movies):
        raise ValueError("movie identifiers must be unique")
    if len(movies) < required_devices:
        raise ValueError("each required CUDA device must receive at least one movie")
    selected_tokens = tokens[:required_devices]
    if len(set(selected_tokens)) != required_devices:
        raise ValueError("CUDA device tokens must be unique")
    assignments = tuple(
        MovieShard(
            shard_index=index,
            shard_count=required_devices,
            cuda_token=selected_tokens[index],
            movie_ids=movies[index::required_devices],
        )
        for index in range(required_devices)
    )
    if any(not shard.movie_ids for shard in assignments):
        raise RuntimeError("a required GPU shard is empty")
    return assignments


def worker_environment(
    shard: MovieShard,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": shard.cuda_token,
            "BIOHUB_SHARD_INDEX": str(shard.shard_index),
            "BIOHUB_SHARD_COUNT": str(shard.shard_count),
            "BIOHUB_SHARD_MOVIES": ",".join(shard.movie_ids),
        }
    )
    return environment


def validate_shard_outputs(
    shards: Sequence[MovieShard],
    observed_by_shard: Mapping[int, Sequence[str]],
) -> tuple[str, ...]:
    """Reject missing, extra, or duplicate movies before submission assembly."""

    expected_all = {movie for shard in shards for movie in shard.movie_ids}
    seen: set[str] = set()
    for shard in shards:
        observed = {str(movie) for movie in observed_by_shard.get(shard.shard_index, ())}
        expected = set(shard.movie_ids)
        if observed != expected:
            raise RuntimeError(
                f"shard {shard.shard_index} output mismatch: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        overlap = seen & observed
        if overlap:
            raise RuntimeError(f"duplicate movies across GPU shards: {sorted(overlap)}")
        seen.update(observed)
    if seen != expected_all:
        raise RuntimeError("merged GPU shards do not cover the planned movie set")
    return tuple(sorted(seen))


def shard_plan_sha256(shards: Sequence[MovieShard]) -> str:
    payload = [asdict(shard) for shard in shards]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
