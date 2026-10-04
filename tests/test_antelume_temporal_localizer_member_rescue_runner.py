from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-temporal-localizer-member-rescue-v1.sh"
PLANNER = ROOT / "research/temporal_localization/plan_member_rescue.py"


def test_rescue_runner_waits_for_original_and_graph_before_fixed_seed_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'while test ! -f "$original_root/run.complete"' in source
    assert "graph-context-division-recovery-v1.sh" in source
    assert "wait_for_stable_idle_gpu" in source
    assert 'test "$idle_polls" -lt 5' in source
    assert 'planner_sha256=__PLANNER_SHA256__' in source
    assert 'plan_status=$(json_field "$final_plan" status)' in source
    assert 'seed=$(json_field "$final_plan" next_seed)' in source
    assert 'gpu_index=$(json_field "$final_plan" next_gpu_index)' in source
    assert "--real-replay-probability 0.25" in source
    assert "--division-critical-per-batch 4" in source
    assert "--member-max-wall-seconds 16500" in source
    assert "score_real_development_probe.py" in source


def test_rescue_never_ranks_members_or_manages_other_workloads() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    planner = PLANNER.read_text(encoding="utf-8")

    assert '"member_or_policy_ranking_performed": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert "RESCUE_SEEDS = (41_057, 41_063, 41_071, 41_081, 41_087, 41_099)" in planner
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "kill " not in source
    assert "kaggle" not in source.lower()
    assert "submit" not in source.lower()


def test_rescue_preserves_original_evidence_and_builds_a_new_archive() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'cp -a "$original_results" "$output_root"' in source
    assert 'mv "$output_root/$name" "$output_root/original-$name"' in source
    assert 'test ! -e "$result_parent"' in source
    assert 'test ! -e "$result_archive"' in source
    assert 'sha256sum "$result_archive" >"$result_archive.sha256"' in source
    assert 'write_terminal "completed" "$final_plan"' in source
