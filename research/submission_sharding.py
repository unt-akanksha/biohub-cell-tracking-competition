"""Mandatory two-GPU movie sharding for future Biohub submission kernels."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


KAGGLE_GPU_NOTEBOOK_MAX_SECONDS = 43_200
MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS = 7_200
DEFAULT_INFERENCE_HARD_STOP_SECONDS = (
    KAGGLE_GPU_NOTEBOOK_MAX_SECONDS - MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS
)
KAGGLE_T4_X2_MACHINE_SHAPE = "NvidiaTeslaT4"
WORKER_TERMINATION_GRACE_SECONDS = 15.0


def validate_inference_hard_stop(seconds: int) -> int:
    """Keep final inference inside Kaggle's 12-hour GPU notebook limit."""

    value = int(seconds)
    if not 0 < value <= DEFAULT_INFERENCE_HARD_STOP_SECONDS:
        raise ValueError(
            "inference hard stop must be positive and no greater than "
            f"{DEFAULT_INFERENCE_HARD_STOP_SECONDS} to preserve the Kaggle "
            "notebook finalization reserve"
        )
    return value


def validate_submission_kernel_metadata(metadata: Mapping[str, object]) -> None:
    """Require the Kaggle T4 x2 request before runtime device verification."""

    expected = {
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": KAGGLE_T4_X2_MACHINE_SHAPE,
    }
    drift = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if drift:
        raise ValueError(f"unsafe Kaggle submission kernel metadata: {drift}")


def terminate_and_reap_processes(
    processes: Sequence[Any], *, grace_seconds: float = WORKER_TERMINATION_GRACE_SECONDS
) -> None:
    """Bound worker shutdown so a timed-out shard cannot outlive its parent."""

    if grace_seconds < 0:
        raise ValueError("process termination grace must be nonnegative")
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in alive:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    survivors = [process for process in alive if process.poll() is None]
    for process in survivors:
        process.kill()
    for process in survivors:
        process.wait()


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
    movie_weights: Mapping[str, float] | None = None,
) -> tuple[MovieShard, ...]:
    """Assign each movie exactly once across the required GPU workers.

    When runtime weights are supplied, longest-processing-time scheduling
    balances whole movies without splitting one movie across workers.  This is
    the production path for Biohub, where movie sizes differ substantially.
    The unweighted round-robin behavior remains deterministic for callers that
    do not yet have a cost inventory.
    """

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
    if movie_weights is None:
        assigned_movies = [list(movies[index::required_devices]) for index in range(required_devices)]
    else:
        normalized_weights = {str(movie): float(weight) for movie, weight in movie_weights.items()}
        if set(normalized_weights) != set(movies):
            raise ValueError("movie weights must cover exactly the planned movies")
        if any(not (weight > 0.0) for weight in normalized_weights.values()):
            raise ValueError("movie weights must be finite and positive")
        if any(weight == float("inf") for weight in normalized_weights.values()):
            raise ValueError("movie weights must be finite and positive")
        assigned_movies = [[] for _ in range(required_devices)]
        assigned_costs = [0.0 for _ in range(required_devices)]
        ranked_movies = sorted(movies, key=lambda movie: (-normalized_weights[movie], movie))
        for movie in ranked_movies:
            shard_index = min(
                range(required_devices),
                key=lambda index: (assigned_costs[index], index),
            )
            assigned_movies[shard_index].append(movie)
            assigned_costs[shard_index] += normalized_weights[movie]
    assignments = tuple(
        MovieShard(
            shard_index=index,
            shard_count=required_devices,
            cuda_token=selected_tokens[index],
            movie_ids=tuple(assigned_movies[index]),
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

    expected_indices = {shard.shard_index for shard in shards}
    if (len(shards) != 2 or expected_indices != {0, 1}
        or any(shard.shard_count != 2 or not shard.movie_ids for shard in shards)):
        raise RuntimeError("invalid two-GPU shard plan")
    if set(observed_by_shard) != expected_indices:
        raise RuntimeError("missing or extra GPU shard outputs")
    expected_all = {movie for shard in shards for movie in shard.movie_ids}
    if sum(len(shard.movie_ids) for shard in shards) != len(expected_all):
        raise RuntimeError("duplicate movies in GPU shard plan")
    seen: set[str] = set()
    for shard in shards:
        observed_rows = tuple(str(movie) for movie in observed_by_shard[shard.shard_index])
        observed = set(observed_rows)
        if len(observed_rows) != len(observed):
            raise RuntimeError(f"duplicate movie outputs within GPU shard {shard.shard_index}")
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
