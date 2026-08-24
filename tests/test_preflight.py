from __future__ import annotations

from datetime import datetime, timezone

import pytest

from biohub_tracker.preflight import (
    BASE_CHECKS,
    LONG_RUN_CHECKS,
    PreflightCheck,
    PreflightError,
    PreflightReport,
    validate_preflight,
    write_preflight_report,
)


NOW = datetime(2026, 8, 24, 0, 0, tzinfo=timezone.utc)


def make_report(tmp_path, runtime: str, *, run_id: str = "run-a"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("passed", encoding="utf-8")
    names = list(BASE_CHECKS)
    if float(runtime) > 1.0:
        names.extend(LONG_RUN_CHECKS)
    checks = [PreflightCheck.create(name, "passed", [evidence], tmp_path) for name in names]
    report = PreflightReport.create(run_id, checks, created_at=NOW)
    path = tmp_path / "preflight.json"
    write_preflight_report(path, report)
    return path, evidence, report


def test_one_hour_does_not_require_dense_or_coverage(tmp_path):
    path, _, _ = make_report(tmp_path, "1.00")
    validated = validate_preflight(
        path, tmp_path, run_id="run-a", declared_max_runtime_hours="1.00", now=NOW
    )
    assert {check.name for check in validated.checks} == set(BASE_CHECKS)


def test_one_point_zero_one_requires_dense_and_coverage(tmp_path):
    path, _, _ = make_report(tmp_path, "1.00")
    with pytest.raises(PreflightError, match="dense_memory"):
        validate_preflight(
            path, tmp_path, run_id="run-a", declared_max_runtime_hours="1.01", now=NOW
        )
    long_path, _, validated_source = make_report(tmp_path / "long", "1.01")
    validated = validate_preflight(
        long_path,
        tmp_path / "long",
        run_id="run-a",
        declared_max_runtime_hours="1.01",
        now=NOW,
    )
    assert validated.report_sha256 == validated_source.report_sha256


def test_changed_evidence_wrong_run_failed_and_stale_reject(tmp_path):
    path, evidence, _ = make_report(tmp_path, "1.00")
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(PreflightError, match="evidence changed"):
        validate_preflight(path, tmp_path, run_id="run-a", declared_max_runtime_hours="1", now=NOW)
    other_path, _, _ = make_report(tmp_path / "other", "1.00")
    with pytest.raises(PreflightError, match="different run"):
        validate_preflight(
            other_path, tmp_path / "other", run_id="run-b", declared_max_runtime_hours="1", now=NOW
        )
    with pytest.raises(PreflightError, match="stale"):
        validate_preflight(
            other_path,
            tmp_path / "other",
            run_id="run-a",
            declared_max_runtime_hours="1",
            now=datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
        )


def test_model_step_may_be_explicitly_not_applicable(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("ok", encoding="utf-8")
    checks = []
    for name in BASE_CHECKS:
        if name == "model_step":
            checks.append(
                PreflightCheck.create(name, "not_applicable", [], tmp_path, detail="inference-only")
            )
        else:
            checks.append(PreflightCheck.create(name, "passed", [evidence], tmp_path))
    report = PreflightReport.create("run-a", checks, created_at=NOW)
    path = write_preflight_report(tmp_path / "preflight.json", report)
    validate_preflight(path, tmp_path, run_id="run-a", declared_max_runtime_hours="1", now=NOW)
