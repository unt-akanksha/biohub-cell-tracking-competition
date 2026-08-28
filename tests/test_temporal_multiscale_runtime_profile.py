from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile-temporal-multiscale-runtime.py"
BASE = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "public-0927-clean-repro-v2"
    / "submission.csv"
)


def test_multiscale_runtime_profile_covers_all_reciprocal_cases() -> None:
    namespace = runpy.run_path(str(SCRIPT))
    result = namespace["profile"](BASE)

    assert result["status"] == "profiled"
    assert result["parameters_per_fold"] == {
        "v3": 20_747_761,
        "v4": 46_386_607,
    }
    assert 1.12 < result["v4_to_v3_macs_ratio"] < 1.14
    assert 2.23 < result["v4_to_v3_parameter_ratio"] < 2.24
    assert set(result["cases"]) == {
        "44b6_1x__6bba_1x",
        "44b6_1x__6bba_2x",
        "44b6_2x__6bba_1x",
        "44b6_2x__6bba_2x",
    }
    assert result["current_scheduler_worst_projected_load_ratio"] < 1.01
    assert result["scheduler_decision"] == (
        "retain_current_transition_plan_no_runtime_mutation"
    )
    assert result["gpu_used"] is False
    assert result["public_leaderboard_used_for_selection"] is False
    assert result["competition_submission_performed"] is False


def test_runtime_profiler_contains_no_external_or_submission_action() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert "kaggle kernels" not in source
    assert "torch.cuda" not in source
