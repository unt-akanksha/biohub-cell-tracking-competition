from __future__ import annotations

import pytest

from research.submission_sharding import (
    build_movie_shards,
    shard_plan_sha256,
    validate_shard_outputs,
    visible_cuda_tokens,
    worker_environment,
)


def test_dual_gpu_plan_assigns_every_movie_once() -> None:
    shards = build_movie_shards(
        ["movie_d", "movie_a", "movie_c", "movie_b"],
        ["GPU-a", "GPU-b"],
    )
    assert [shard.movie_ids for shard in shards] == [
        ("movie_d", "movie_c"),
        ("movie_a", "movie_b"),
    ]
    assert {shard.cuda_token for shard in shards} == {"GPU-a", "GPU-b"}
    assert len(shard_plan_sha256(shards)) == 64


def test_submission_plan_refuses_single_gpu_or_duplicate_movies() -> None:
    with pytest.raises(RuntimeError, match="requires 2"):
        build_movie_shards(["a", "b"], ["0"])
    with pytest.raises(ValueError, match="unique"):
        build_movie_shards(["a", "a"], ["0", "1"])


def test_visible_tokens_preserve_parent_device_mapping() -> None:
    assert visible_cuda_tokens(detected_devices=2, environ={}) == ("0", "1")
    assert visible_cuda_tokens(
        detected_devices=2,
        environ={"CUDA_VISIBLE_DEVICES": "GPU-abcd,GPU-efgh"},
    ) == ("GPU-abcd", "GPU-efgh")
    assert visible_cuda_tokens(
        detected_devices=0, environ={"CUDA_VISIBLE_DEVICES": "-1"}
    ) == ()


def test_worker_environment_binds_one_physical_device() -> None:
    shard = build_movie_shards(["a", "b"], ["4", "7"])[1]
    environment = worker_environment(shard, base={"KEEP": "yes"})
    assert environment == {
        "KEEP": "yes",
        "CUDA_VISIBLE_DEVICES": "7",
        "BIOHUB_SHARD_INDEX": "1",
        "BIOHUB_SHARD_COUNT": "2",
        "BIOHUB_SHARD_MOVIES": "b",
    }


def test_output_validation_rejects_wrong_or_duplicate_coverage() -> None:
    shards = build_movie_shards(["a", "b", "c", "d"], ["0", "1"])
    assert validate_shard_outputs(shards, {0: ["a", "c"], 1: ["b", "d"]}) == (
        "a",
        "b",
        "c",
        "d",
    )
    with pytest.raises(RuntimeError, match="output mismatch"):
        validate_shard_outputs(shards, {0: ["a"], 1: ["b", "d"]})
