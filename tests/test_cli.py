from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_tracker.cli import main
from biohub_tracker.kaggle import FixtureRunner
from biohub_tracker.launch import LaunchError, authorize_launch, push_kernel, validate_authorization
from biohub_tracker.ledger import EventType, ExperimentEvent, Ledger, registration_payload
from biohub_tracker.preflight import (
    BASE_CHECKS,
    LONG_RUN_CHECKS,
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)


NOW = datetime(2026, 8, 24, 0, 0, tzinfo=timezone.utc)


def setup_launch_workspace(tmp_path: Path, runtime: str = "2.00"):
    (tmp_path / "config").mkdir()
    config = {
        "slug": "biohub-cell-tracking-during-development",
        "gpu_reserve_hours": "8.00",
        "notebook_runtime_limit_hours": "12.00",
    }
    (tmp_path / "config" / "competition.json").write_text(json.dumps(config), encoding="utf-8")
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "quota.json").write_text(json.dumps([{
        "resource": "GPU", "used": "0.00h", "remaining": "30.00h", "total": "30.00h", "refreshAt": "2026-08-29T00:00:00"
    }]), encoding="utf-8")
    (fixture / "kernels.json").write_text("[]", encoding="utf-8")
    (fixture / "kernel_status.json").write_text("{}", encoding="utf-8")
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": "owner/biohub-job", "enable_gpu": True, "code_file": "main.py"
    }), encoding="utf-8")
    (kernel / "main.py").write_text("print('bounded fixture')\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("passed", encoding="utf-8")
    checks = [
        PreflightCheck.create(name, "passed", [evidence], tmp_path)
        for name in (*BASE_CHECKS, *LONG_RUN_CHECKS)
    ]
    report = PreflightReport.create("run-a", checks, created_at=NOW)
    report_path = write_preflight_report(tmp_path / "preflight.json", report)
    ledger = Ledger(tmp_path / "experiments" / "events.jsonl", tmp_path)
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, registration_payload(
        hypothesis="launch test", parent=None, config={}, seeds=[1], split="test",
        declared_max_runtime_hours=runtime, code={"git_head": "abc"},
    )))
    return config, fixture, kernel, evidence, report_path, ledger


def authorize_fixture(tmp_path, runtime="2.00"):
    config, fixture, kernel, evidence, report_path, ledger = setup_launch_workspace(tmp_path, runtime)
    authorization, decision = authorize_launch(
        workspace_root=tmp_path,
        ledger=ledger,
        runner=FixtureRunner(fixture),
        config=config,
        run_id="run-a",
        kernel_directory=kernel,
        kernel_ref="owner/biohub-job",
        preflight_report=report_path,
        now=NOW,
    )
    assert decision.authorized
    return authorization, config, fixture, kernel, evidence, ledger


def test_launch_without_execute_never_invokes_runner(tmp_path):
    authorization, config, fixture, _, _, ledger = authorize_fixture(tmp_path)
    calls = []
    result = push_kernel(
        authorization,
        workspace_root=tmp_path,
        ledger=ledger,
        runner=FixtureRunner(fixture),
        config=config,
        nonce=authorization.nonce,
        execute=False,
        push_runner=lambda command: calls.append(command),
        now=NOW,
    )
    assert result["executed"] is False and calls == []


def test_launch_fake_runner_exact_command_records_start_and_is_single_use(tmp_path):
    authorization, config, fixture, kernel, _, ledger = authorize_fixture(tmp_path)
    calls = []
    result = push_kernel(
        authorization,
        workspace_root=tmp_path,
        ledger=ledger,
        runner=FixtureRunner(fixture),
        config=config,
        nonce=authorization.nonce,
        execute=True,
        push_runner=lambda command: calls.append(list(command)) or SimpleNamespace(returncode=0),
        now=NOW,
    )
    assert calls == [["kaggle", "kernels", "push", "-p", str(kernel.resolve())]]
    assert result["executed"] is True
    assert ledger.read_events()[-1].event_type is EventType.STARTED
    with pytest.raises(LaunchError) as reused:
        validate_authorization(
            authorization, workspace_root=tmp_path, ledger=ledger,
            runner=FixtureRunner(fixture), config=config, nonce=authorization.nonce, now=NOW,
        )
    assert reused.value.reason_code == "AUTHORIZATION_CONSUMED"


@pytest.mark.parametrize("case", ["expired", "wrong-run", "wrong-path", "wrong-nonce"])
def test_launch_authorization_identity_rejections(tmp_path, case):
    authorization, config, fixture, _, _, ledger = authorize_fixture(tmp_path)
    kwargs = {
        "authorization": authorization,
        "workspace_root": tmp_path,
        "ledger": ledger,
        "runner": FixtureRunner(fixture),
        "config": config,
        "nonce": authorization.nonce,
        "now": NOW,
    }
    if case == "expired":
        kwargs["now"] = datetime(2026, 8, 24, 1, 0, tzinfo=timezone.utc)
    elif case == "wrong-run":
        kwargs["expected_run_id"] = "run-b"
    elif case == "wrong-path":
        other = tmp_path / "other"
        other.mkdir()
        kwargs["expected_kernel_directory"] = other
    else:
        kwargs["nonce"] = "wrong"
    with pytest.raises(LaunchError):
        validate_authorization(**kwargs)


def test_launch_changed_quota_active_kernel_and_preflight_reject(tmp_path):
    authorization, config, fixture, _, evidence, ledger = authorize_fixture(tmp_path)
    (fixture / "quota.json").write_text(json.dumps([{
        "resource": "GPU", "used": "1.00h", "remaining": "29.00h", "total": "30.00h", "refreshAt": "2026-08-29T00:00:00"
    }]), encoding="utf-8")
    with pytest.raises(LaunchError) as changed:
        validate_authorization(
            authorization, workspace_root=tmp_path, ledger=ledger,
            runner=FixtureRunner(fixture), config=config, nonce=authorization.nonce, now=NOW,
        )
    assert changed.value.reason_code == "KAGGLE_STATE_CHANGED"
    (fixture / "quota.json").write_text(json.dumps([{
        "resource": "GPU", "used": "0.00h", "remaining": "30.00h", "total": "30.00h", "refreshAt": "2026-08-29T00:00:00"
    }]), encoding="utf-8")
    (fixture / "kernels.json").write_text(json.dumps([{"ref": "owner/active"}]), encoding="utf-8")
    (fixture / "kernel_status.json").write_text(json.dumps({"owner/active": "RUNNING"}), encoding="utf-8")
    with pytest.raises(LaunchError) as active:
        validate_authorization(
            authorization, workspace_root=tmp_path, ledger=ledger,
            runner=FixtureRunner(fixture), config=config, nonce=authorization.nonce, now=NOW,
        )
    assert active.value.reason_code == "ACTIVE_GPU_KERNEL"
    (fixture / "kernels.json").write_text("[]", encoding="utf-8")
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_authorization(
            authorization, workspace_root=tmp_path, ledger=ledger,
            runner=FixtureRunner(fixture), config=config, nonce=authorization.nonce, now=NOW,
        )


def test_launch_cli_preview_does_not_call_runner(tmp_path, capsys):
    authorization, _, fixture, _, _, _ = authorize_fixture(tmp_path)
    calls = []
    assert main([
        "--root", str(tmp_path), "launch", "execute",
        "--authorization-id", authorization.authorization_id,
        "--fixture-dir", str(fixture),
    ], launch_runner=lambda command: calls.append(command)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["executed"] is False and calls == []


def test_launch_failed_runner_is_consumed_and_audited(tmp_path):
    authorization, config, fixture, _, _, ledger = authorize_fixture(tmp_path)
    with pytest.raises(LaunchError) as failure:
        push_kernel(
            authorization,
            workspace_root=tmp_path,
            ledger=ledger,
            runner=FixtureRunner(fixture),
            config=config,
            nonce=authorization.nonce,
            execute=True,
            push_runner=lambda command: (_ for _ in ()).throw(RuntimeError("fixture push failed")),
            now=NOW,
        )
    assert failure.value.reason_code == "KERNEL_PUSH_FAILED"
    assert ledger.read_events()[-1].event_type is EventType.LAUNCH_FAILED


def test_launch_nonzero_exit_records_redacted_push_diagnostic(tmp_path):
    authorization, config, fixture, _, _, ledger = authorize_fixture(tmp_path)
    result = SimpleNamespace(returncode=1, stderr="API token=do-not-record-this", stdout="")
    with pytest.raises(LaunchError) as failure:
        push_kernel(
            authorization,
            workspace_root=tmp_path,
            ledger=ledger,
            runner=FixtureRunner(fixture),
            config=config,
            nonce=authorization.nonce,
            execute=True,
            push_runner=lambda command: result,
            now=NOW,
        )
    assert failure.value.reason_code == "KERNEL_PUSH_FAILED"
    reason = ledger.read_events()[-1].payload["reason"]
    assert "token=[REDACTED]" in reason
    assert "do-not-record-this" not in reason


def test_launch_consumption_race_invokes_exactly_one_push(tmp_path, monkeypatch):
    import biohub_tracker.launch as launch_module

    authorization, config, fixture, _, _, ledger = authorize_fixture(tmp_path)
    original_validate = launch_module.validate_authorization
    barrier = threading.Barrier(2)
    calls = []
    results = []
    errors = []

    def synchronized_validate(*args, **kwargs):
        value = original_validate(*args, **kwargs)
        barrier.wait()
        return value

    monkeypatch.setattr(launch_module, "validate_authorization", synchronized_validate)

    def attempt():
        try:
            results.append(
                push_kernel(
                    authorization,
                    workspace_root=tmp_path,
                    ledger=ledger,
                    runner=FixtureRunner(fixture),
                    config=config,
                    nonce=authorization.nonce,
                    execute=True,
                    push_runner=lambda command: calls.append(command)
                    or SimpleNamespace(returncode=0),
                    now=NOW,
                )
            )
        except Exception as exc:  # exact loser type asserted below
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(calls) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], LaunchError)
    assert errors[0].reason_code == "AUTHORIZATION_CONSUMED"
