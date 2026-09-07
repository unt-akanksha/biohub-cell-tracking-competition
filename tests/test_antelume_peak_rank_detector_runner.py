from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run-antelume-peak-rank-detector-v1.sh"
)


def test_runner_waits_for_graph_and_gpu_before_training() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "biohub-graph-context-recovery-v1/run.complete" in source
    assert "--query-compute-apps=pid" in source
    assert "sleep 30" in source
    assert "grep -qi A10G" in source
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "kill " not in source


def test_runner_uses_clean_data_and_fixed_training_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "data/synthetic256" in source
    assert "data/real-replay" in source
    assert "--steps 12000" in source
    assert "--widths 96,192,384,768" in source
    assert "--depths 3,3,9,3" in source
    assert "--real-frequency 4" in source
    assert "competition" not in source
    assert "submission" not in source
    assert "public" not in source


def test_runner_hashes_code_data_and_archive() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "real_manifest_sha256=" in source
    assert "trainer_sha256=" in source
    assert "model_sha256=" in source
    assert "objectives_sha256=" in source
    assert "synthetic_data_sha256=" in source
    assert source.count("sha256sum -c -") >= 5
    assert "SHA256SUMS" in source
    assert 'sha256sum "$result_archive"' in source
