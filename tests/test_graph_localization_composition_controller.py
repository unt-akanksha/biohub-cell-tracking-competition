from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-build-verify-submit-graph-localization-composition.ps1"


def test_controller_requires_two_promotions_and_composition_gain() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "graph-context-recovery-consensus-candidate-promotion-version" in source
    assert "temporal-localization-candidate-promotion.json" in source
    assert source.count('status -ne "eligible_for_submission"') >= 1
    assert "independent_component_promotion_required = $true" in source
    assert "verify-graph-localization-composition-candidate.py" in source
    assert "composition_gain_over_best_component" in source


def test_controller_is_private_dual_gpu_hash_bound_and_fail_closed() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert 'expected_gpu_count = 2' in source
    assert 'machine_shape = "NvidiaTeslaT4"' in source
    assert '$metadata.is_private -ne $true' in source
    assert '$metadata.enable_internet -ne $false' in source
    assert "Get-FileHash -Algorithm SHA256" in source
    assert "public_leaderboard_used_for_selection = $false" in source
    assert '"composition_rejected"' in source
    assert "submit-graph-localization-composition-candidate.py" in source
    assert source.count("competitions submit") == 1


def test_controller_has_no_aws_or_rsna_control_surface() -> None:
    source = CONTROLLER.read_text(encoding="utf-8").lower()
    assert "rsna" not in source
    assert "systemctl" not in source
    assert "nvidia-smi" not in source
    assert "aws " not in source

