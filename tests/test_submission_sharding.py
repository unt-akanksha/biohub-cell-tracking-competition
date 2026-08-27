from __future__ import annotations

import subprocess

import pytest

from research.submission_sharding import (
    DEFAULT_INFERENCE_HARD_STOP_SECONDS,
    KAGGLE_T4_X2_MACHINE_SHAPE,
    KAGGLE_GPU_NOTEBOOK_MAX_SECONDS,
    MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS,
    WORKER_TERMINATION_GRACE_SECONDS,
    build_movie_shards,
    shard_plan_sha256,
    terminate_and_reap_processes,
    validate_inference_hard_stop,
    validate_submission_kernel_metadata,
    validate_shard_outputs,
    visible_cuda_tokens,
    worker_environment,
)


def test_submission_runtime_policy_preserves_two_hours() -> None:
    assert DEFAULT_INFERENCE_HARD_STOP_SECONDS == 36_000
    assert (
        KAGGLE_GPU_NOTEBOOK_MAX_SECONDS - DEFAULT_INFERENCE_HARD_STOP_SECONDS
        == MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS
        == 7_200
    )
    assert validate_inference_hard_stop(36_000) == 36_000
    with pytest.raises(ValueError, match="finalization reserve"):
        validate_inference_hard_stop(36_001)


def test_submission_metadata_requires_offline_t4_x2_shape() -> None:
    metadata = {
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": KAGGLE_T4_X2_MACHINE_SHAPE,
    }
    validate_submission_kernel_metadata(metadata)

    with pytest.raises(ValueError, match="unsafe Kaggle"):
        validate_submission_kernel_metadata({**metadata, "enable_internet": True})
    with pytest.raises(ValueError, match="unsafe Kaggle"):
        validate_submission_kernel_metadata(
            {**metadata, "machine_shape": "NvidiaTeslaP100"}
        )


class FakeProcess:
    def __init__(self, *, exits_on_terminate: bool) -> None:
        self.return_code = None
        self.exits_on_terminate = exits_on_terminate
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if self.exits_on_terminate:
            self.return_code = -15

    def wait(self, timeout=None):
        self.waited = True
        if self.return_code is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
            raise AssertionError("unbounded wait occurred before force kill")
        return self.return_code

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9


def test_timed_out_workers_are_terminated_force_killed_and_reaped() -> None:
    assert WORKER_TERMINATION_GRACE_SECONDS == 15.0
    cooperative = FakeProcess(exits_on_terminate=True)
    stuck = FakeProcess(exits_on_terminate=False)

    terminate_and_reap_processes([cooperative, stuck], grace_seconds=0)

    assert cooperative.terminated and cooperative.waited and not cooperative.killed
    assert stuck.terminated and stuck.killed and stuck.waited


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


def test_weighted_plan_balances_large_whole_movies_deterministically() -> None:
    movies = ["large", "medium", "small_a", "small_b"]
    weights = {"large": 10, "medium": 6, "small_a": 2, "small_b": 2}

    shards = build_movie_shards(movies, ["0", "1"], movie_weights=weights)

    assert shards[0].movie_ids == ("large",)
    assert shards[1].movie_ids == ("medium", "small_a", "small_b")
    assert {
        movie for shard in shards for movie in shard.movie_ids
    } == set(movies)


def test_weighted_plan_requires_exact_positive_cost_inventory() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        build_movie_shards(["a", "b"], ["0", "1"], movie_weights={"a": 1})
    with pytest.raises(ValueError, match="finite and positive"):
        build_movie_shards(
            ["a", "b"], ["0", "1"], movie_weights={"a": 1, "b": 0}
        )


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
