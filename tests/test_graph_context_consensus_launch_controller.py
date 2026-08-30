from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-launch-graph-context-consensus-candidate.ps1"
GENERIC = ROOT / "scripts/wait-launch-relational-consensus-candidate.ps1"


def test_launch_binds_graph_context_runtime_kernel_and_promotion_tools() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "biohub-graph-context-consensus-division-v1" in source
    assert "biohub-ema-graph-context-consensus-v1" in source
    assert "build-graph-context-consensus-submission-candidate.py" in source
    assert "verify-graph-context-consensus-submission-candidate.py" in source
    assert "submit-graph-context-consensus-candidate.py" in source
    assert "graph-context-consensus-candidate-controller-v1" in source


def test_shared_launch_waits_for_scientific_gate_and_requires_two_t4s() -> None:
    source = GENERIC.read_text(encoding="utf-8")

    assert 'status -ne "runtime_packaged"' in source
    assert 'runtime_created -ne $true' in source
    assert 'authorized_for_full_candidate_evaluation -ne $true' in source
    assert 'expected_gpu_count = 2' in source
    assert 'machine_shape = "NvidiaTeslaT4"' in source
    assert "Exactly two T4 GPUs are required" in source
    assert '"cuda:0"' in source and '"cuda:1"' in source
    assert '"ThreadPoolExecutor"' in source
    assert "kaggle competitions submit" not in source
