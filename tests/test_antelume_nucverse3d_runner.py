from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run-antelume-nucverse3d-compatibility-v1.sh"
)


def test_runner_is_sequential_hash_bound_and_bounded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "capacity-pu-v3/run.complete" in source
    assert "while test ! -f \"$v3_complete\"" in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "minimum_free_bytes=1300000000" in source
    assert "timeout --signal=TERM --kill-after=30s 7200s" in source
    assert "--phase optimization" in source
    assert "--phase selection" in source
    assert '--optimization-receipt "$result_root/optimization-screen.json"' in source
    assert 'report.get("compatibility_passed") is True' in source
    assert "remaining_seconds=$((7200 - elapsed_seconds))" in source
    assert "--per-embryo 8" in source
    assert "--maximum-points-per-example 4" in source
    assert "competition" not in source.lower().split("real-replay", 1)[-1]

    expected = {
        "screen_sha256": "9d696b327472d420852ca3122e89a9558f725ac2b2c1a13bed5e8e33ab9121a4",
        "onnx_sha256": "ca16e1b26d21ae522d68aba384ee7121f5ba2de28a122c291d1e1e627601e871",
        "onnx_wheel_sha256": "4f3fb5cc4e2898ac5312a7dc03a65133dd2abf9a5e520e69afb880a7251ec97a",
        "onnx2torch_wheel_sha256": "123258e0f147e07b259cf845c8113c5634b8260e52c5fb26b7d507226732d9e5",
        "manifest_sha256": "5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697",
    }
    for key, value in expected.items():
        assert f"{key}={value}" in source


def test_runner_cannot_submit_or_touch_unrelated_workloads() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "kaggle competitions submit" not in source
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "rsna" not in source.lower()
    assert "rm -rf" not in source
