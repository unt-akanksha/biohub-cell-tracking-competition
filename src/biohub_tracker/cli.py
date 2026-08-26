from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .guard import GuardInputError, evaluate_guard, list_active_gpu_kernels, read_gpu_quota
from .kaggle import FixtureRunner, KaggleRunner
from .launch import authorize_launch, default_push_runner, load_authorization, push_kernel

from .ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    LedgerError,
    amendment_payload,
    artifact_record,
    completed_payload,
    cpu_acceptance_completed_payload,
    cpu_acceptance_failed_payload,
    cpu_acceptance_inputs_bound_payload,
    cpu_acceptance_started_payload,
    decision_payload,
    event_sha256,
    failed_payload,
    generate_run_id,
    git_state,
    rejected_payload,
    registration_payload,
    reconstruct_cpu_acceptances,
    reconstruct_runs,
    start_payload,
)
from .progress import render_progress_json, render_progress_markdown, write_progress_reports
from .preflight import validate_preflight
from .watch import collect_snapshot, persist_snapshot, status_line, write_status_reports


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_value(value: str | None) -> dict:
    if value is None:
        return {}
    candidate = Path(value)
    if not value.lstrip().startswith("{") and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("configuration must be a JSON object")
    return parsed


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_float=Decimal)
    if not isinstance(value, dict):
        raise ValueError("metrics report must be a JSON object")
    return value


def _artifact_records(root: Path, paths: Sequence[Path] | None) -> list[dict[str, str]]:
    return [artifact_record(root, path) for path in (paths or [])]


def _rooted(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _prediction_refs(root: Path, values: Sequence[str]):
    from .evaluation import PredictionSetRef

    result = []
    for value in values:
        fold, separator, raw_path = value.partition("=")
        if not separator or not fold.strip() or not raw_path.strip():
            raise ValueError("prediction sets must use FOLD_ID=GRAPH_DIRECTORY")
        directory = _rooted(root, Path(raw_path))
        result.append(
            PredictionSetRef(
                fold_id=fold.strip(),
                graph_dir=directory,
                producer_manifest_path=directory / "prediction-set.json",
            )
        )
    result.sort(key=lambda item: item.fold_id)
    if len(result) != 2 or len({item.fold_id for item in result}) != 2:
        raise ValueError("each role requires exactly two distinct reciprocal folds")
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biohub", description="Biohub competition control plane")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    watch = subparsers.add_parser("watch", help="capture read-only competition state")
    source = watch.add_mutually_exclusive_group()
    source.add_argument("--live", action="store_true", help="read the authenticated Kaggle CLI")
    source.add_argument("--fixture-dir", type=Path, help="read deterministic JSON fixtures")
    watch.add_argument("--top", type=int, default=20, help="number of public notebooks to inspect")
    watch.add_argument(
        "--audit-notebook-sources",
        action="store_true",
        help="pull and statically inspect notebook source without executing it",
    )

    experiment = subparsers.add_parser("experiment", help="append immutable experiment events")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    register = experiment_commands.add_parser("register", help="register an experiment")
    register.add_argument("--run-id")
    register.add_argument("--hypothesis", required=True)
    register.add_argument("--max-runtime-hours", required=True)
    register.add_argument("--parent")
    register.add_argument("--config")
    register.add_argument("--seed", action="append", type=int, default=[])
    register.add_argument("--split", default="not-registered")
    register.add_argument("--data-path", type=Path)
    register.add_argument("--data-sha256")
    register.add_argument("--model-path", type=Path)
    register.add_argument("--model-sha256")
    register.add_argument("--producer-evidence", help="JSON object/file with immutable lineage")
    repair = experiment_commands.add_parser("repair", help="acknowledge a quarantined tail")
    repair.add_argument("--reason", required=True)
    start = experiment_commands.add_parser("start", help="record a launched experiment")
    start.add_argument("run_id")
    start.add_argument("--kaggle-ref", required=True)
    start.add_argument("--authorization-id", required=True)
    start.add_argument("--quota-before-hours", required=True)
    finish = experiment_commands.add_parser("finish", help="record a completed experiment")
    finish.add_argument("run_id")
    finish.add_argument("--actual-runtime-hours", required=True)
    finish.add_argument("--quota-after-hours", required=True)
    finish.add_argument("--metrics-report", type=Path, required=True)
    finish.add_argument("--artifact", type=Path, action="append", default=[])
    finish.add_argument("--report", type=Path, action="append", default=[])
    finish.add_argument("--public-score")
    finish.add_argument("--producer-evidence", help="JSON object/file with terminal inventory evidence")
    fail = experiment_commands.add_parser("fail", help="record an experiment failure")
    fail.add_argument("run_id")
    fail.add_argument("--actual-runtime-hours", required=True)
    fail.add_argument("--quota-after-hours", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--traceback", type=Path)
    reject = experiment_commands.add_parser("reject", help="record a failed experiment gate")
    reject.add_argument("run_id")
    reject.add_argument("--actual-runtime-hours", required=True)
    reject.add_argument("--quota-after-hours", required=True)
    reject.add_argument("--gate", required=True)
    reject.add_argument("--reason", required=True)
    decide = experiment_commands.add_parser("decide", help="record a promotion decision")
    decide.add_argument("run_id")
    decide.add_argument(
        "--decision",
        choices=["promote", "retain", "retire", "inconclusive"],
        required=True,
    )
    decide.add_argument("--evidence", action="append", required=True)
    amend = experiment_commands.add_parser("amend", help="append a correction without mutation")
    amend.add_argument("run_id")
    amend.add_argument("--target-event-id", required=True)
    amend.add_argument("--reason", required=True)
    amend.add_argument("--replacement", required=True)
    progress = subparsers.add_parser("progress", help="render immutable experiment lineage")
    progress.add_argument("--json", action="store_true", dest="json_output")
    guard = subparsers.add_parser("guard", help="evaluate quota-safe Kaggle launch eligibility")
    guard.add_argument("--run-id", required=True)
    guard.add_argument("--max-runtime-hours", required=True)
    guard_source = guard.add_mutually_exclusive_group(required=True)
    guard_source.add_argument("--live", action="store_true", help="read authenticated Kaggle state")
    guard_source.add_argument("--fixture-dir", type=Path, help="read deterministic fixtures")
    guard.add_argument("--json", action="store_true", dest="json_output")
    preflight = subparsers.add_parser("preflight", help="validate hashed preflight evidence")
    preflight_commands = preflight.add_subparsers(dest="preflight_command", required=True)
    preflight_validate = preflight_commands.add_parser("validate", help="validate a report")
    preflight_validate.add_argument("--run-id", required=True)
    preflight_validate.add_argument("--max-runtime-hours", required=True)
    preflight_validate.add_argument("--report", type=Path, required=True)
    preflight_validate.add_argument("--max-age-seconds", type=int, default=3600)
    launch = subparsers.add_parser("launch", help="authorize or execute a guarded Kaggle launch")
    launch_commands = launch.add_subparsers(dest="launch_command", required=True)
    authorize = launch_commands.add_parser("authorize", help="create a short-lived authorization")
    authorize.add_argument("--run-id", required=True)
    authorize.add_argument("--kernel-dir", type=Path, required=True)
    authorize.add_argument("--kernel-ref", required=True)
    authorize.add_argument("--preflight-report", type=Path, required=True)
    authorize.add_argument("--ttl-seconds", type=int, default=600)
    authorize_source = authorize.add_mutually_exclusive_group(required=True)
    authorize_source.add_argument("--live", action="store_true")
    authorize_source.add_argument("--fixture-dir", type=Path)
    execute = launch_commands.add_parser("execute", help="consume an authorization and push")
    execute.add_argument("--authorization-id", required=True)
    execute.add_argument("--nonce")
    execute.add_argument("--execute", action="store_true", help="actually invoke the push runner")
    execute_source = execute.add_mutually_exclusive_group()
    execute_source.add_argument("--live", action="store_true")
    execute_source.add_argument("--fixture-dir", type=Path)
    scorer = subparsers.add_parser("scorer", help="verify the pinned official scorer")
    scorer_commands = scorer.add_subparsers(dest="scorer_command", required=True)
    scorer_verify = scorer_commands.add_parser("verify", help="verify provenance and run tracer")
    scorer_verify.add_argument("--lock", type=Path)
    scorer_verify.add_argument("--checkout", type=Path)
    scorer_verify.add_argument("--tracksdata-checkout", type=Path)
    scorer_verify.add_argument("--fixture", type=Path)
    scorer_verify.add_argument("--expected", type=Path)
    scorer_verify.add_argument("--live", action="store_true")
    manifest = subparsers.add_parser("manifest", help="build or verify reciprocal manifests")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_build = manifest_commands.add_parser("build", help="build an immutable manifest")
    manifest_build.add_argument("--data-root", type=Path, required=True)
    manifest_build.add_argument("--output", type=Path, required=True)
    manifest_build.add_argument("--scorer-lock", type=Path)
    manifest_verify = manifest_commands.add_parser("verify", help="verify a frozen manifest")
    manifest_verify.add_argument("--manifest", type=Path, required=True)
    manifest_verify.add_argument("--data-root", type=Path)
    manifest_verify.add_argument("--scorer-lock", type=Path)
    graph = subparsers.add_parser("graph", help="validate ledger-bound prediction graphs")
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    graph_validate = graph_commands.add_parser("validate", help="validate a complete prediction set")
    graph_validate.add_argument("--pred-dir", type=Path, required=True)
    graph_validate.add_argument("--producer-manifest", type=Path, required=True)
    graph_validate.add_argument("--manifest", type=Path, required=True)
    graph_validate.add_argument("--fold", required=True)
    graph_validate.add_argument("--ledger", type=Path, required=True)
    graph_validate.add_argument("--mode", choices=["native", "submission"], required=True)
    graph_validate.add_argument("--scorer-lock", type=Path)
    graph_validate.add_argument("--scorer-checkout", type=Path)
    graph_validate.add_argument("--tracksdata-checkout", type=Path)
    submission = subparsers.add_parser("submission", help="build authoritative submission space")
    submission_commands = submission.add_subparsers(dest="submission_command", required=True)
    roundtrip = submission_commands.add_parser("roundtrip", help="project CSV and rebuild GEFF")
    roundtrip.add_argument("--pred-dir", type=Path, required=True)
    roundtrip.add_argument("--producer-manifest", type=Path, required=True)
    roundtrip.add_argument("--manifest", type=Path, required=True)
    roundtrip.add_argument("--fold", required=True)
    roundtrip.add_argument("--ledger", type=Path, required=True)
    roundtrip.add_argument("--truth-root", type=Path, required=True)
    roundtrip.add_argument("--output-dir", type=Path, required=True)
    roundtrip.add_argument("--scorer-lock", type=Path)
    roundtrip.add_argument("--scorer-checkout", type=Path)
    roundtrip.add_argument("--tracksdata-checkout", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="run ledger-bound exact evaluation")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_exact = evaluate_commands.add_parser("exact", help="run four-set reciprocal scoring")
    evaluate_exact.add_argument("--ledger", type=Path, required=True)
    evaluate_exact.add_argument("--evaluation-run-id", required=True)
    evaluate_exact.add_argument("--truth-dir", type=Path, required=True)
    evaluate_exact.add_argument("--manifest", type=Path, required=True)
    evaluate_exact.add_argument("--scorer-lock", type=Path, required=True)
    evaluate_exact.add_argument("--evaluation-policy", type=Path, required=True)
    evaluate_exact.add_argument("--baseline-set", action="append", required=True)
    evaluate_exact.add_argument("--candidate-set", action="append", required=True)
    evaluate_exact.add_argument("--output-dir", type=Path, required=True)
    evaluate_exact.add_argument("--scorer-checkout", type=Path)
    evaluate_exact.add_argument("--tracksdata-checkout", type=Path)
    promote = subparsers.add_parser("promote", help="evaluate exact candidate promotion")
    promote_commands = promote.add_subparsers(dest="promote_command", required=True)
    promote_evaluate = promote_commands.add_parser("evaluate", help="evaluate or record policy")
    promote_evaluate.add_argument("--ledger", type=Path, required=True)
    promote_evaluate.add_argument("--report", type=Path, required=True)
    promote_evaluate.add_argument("--policy", type=Path, required=True)
    promote_evaluate.add_argument("--evaluation-run-id", required=True)
    promote_evaluate.add_argument("--record", action="store_true")
    cpu_acceptance = subparsers.add_parser(
        "cpu-acceptance", help="manage the isolated CPU official-data control lifecycle"
    )
    cpu_commands = cpu_acceptance.add_subparsers(
        dest="cpu_acceptance_command", required=True
    )
    cpu_register = cpu_commands.add_parser("register", help="pre-register a request")
    cpu_register_source = cpu_register.add_mutually_exclusive_group(required=True)
    cpu_register_source.add_argument(
        "--request", type=Path, help="verify a canonical already-registered request"
    )
    cpu_register_source.add_argument("--config", type=Path)
    cpu_register.add_argument("--run-id")
    cpu_register.add_argument("--evaluation-run-id")
    cpu_register.add_argument("--kernel-ref")
    cpu_register.add_argument("--runtime-dataset-ref")
    cpu_register.add_argument("--runtime-bundle", type=Path)
    cpu_register.add_argument("--output", type=Path)
    cpu_bundle = cpu_commands.add_parser(
        "build-bundle", help="build one deterministic opaque runtime bundle"
    )
    cpu_bundle.add_argument("--source", type=Path, required=True)
    cpu_bundle.add_argument("--output", type=Path, required=True)
    cpu_verify_bundle = cpu_commands.add_parser(
        "verify-bundle", help="securely extract and verify an opaque runtime bundle"
    )
    cpu_verify_bundle.add_argument("--bundle", type=Path, required=True)
    cpu_verify_bundle.add_argument("--destination", type=Path, required=True)
    cpu_preflight = cpu_commands.add_parser(
        "preflight", help="reject any nonterminal CPU or exact acceptance lifecycle"
    )
    cpu_start = cpu_commands.add_parser("start", help="bind owned asset versions")
    cpu_start.add_argument("run_id")
    cpu_start.add_argument("--kernel-ref", required=True)
    cpu_start.add_argument("--runtime-dataset-ref", required=True)
    cpu_reconcile = cpu_commands.add_parser(
        "reconcile", help="append validated input and terminal reconciliation evidence"
    )
    cpu_reconcile.add_argument("run_id")
    cpu_reconcile.add_argument("--evidence", type=Path, required=True)
    cpu_reconcile.add_argument("--quota-before", type=Path, required=True)
    cpu_reconcile.add_argument("--quota-after", type=Path, required=True)
    cpu_reconcile.add_argument(
        "--manifest-output", type=Path, default=Path("manifests/reciprocal-embryo-v1.json")
    )
    cpu_reconcile.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/exact/phase2-control-acceptance.json"),
    )
    cpu_fail = cpu_commands.add_parser("fail", help="terminate a nonterminal CPU control")
    cpu_fail.add_argument("run_id")
    cpu_fail.add_argument("--reason-code", required=True)
    cpu_fail.add_argument("--detail", required=True)
    cpu_fail.add_argument("--observed-artifacts")
    cpu_fail.add_argument("--evaluation-run-id")
    return parser


def _main(argv: Sequence[str] | None = None, *, launch_runner=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "watch":
        fixture_dir = args.fixture_dir
        if not args.live and fixture_dir is None:
            fixture_dir = root / "tests" / "fixtures" / "kaggle"
        config = _load_config(root / "config" / "competition.json")
        snapshot = collect_snapshot(
            config,
            fixture_dir=fixture_dir,
            live=args.live,
            notebook_limit=args.top,
            root=root,
            audit_notebook_sources=args.audit_notebook_sources,
        )
        path = persist_snapshot(snapshot, root)
        report_path, _ = write_status_reports(snapshot, config, root)
        print(path)
        print(report_path)
        print(status_line(snapshot))
        return 0
    if args.command == "experiment":
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        if args.experiment_command == "register":
            run_id = args.run_id or generate_run_id(args.hypothesis)
            data = (
                artifact_record(root, args.data_path, expected_sha256=args.data_sha256)
                if args.data_path
                else ({"sha256": args.data_sha256} if args.data_sha256 else None)
            )
            model = (
                artifact_record(root, args.model_path, expected_sha256=args.model_sha256)
                if args.model_path
                else ({"sha256": args.model_sha256} if args.model_sha256 else None)
            )
            payload = registration_payload(
                hypothesis=args.hypothesis,
                parent=args.parent,
                config=_json_value(args.config),
                seeds=args.seed,
                split=args.split,
                declared_max_runtime_hours=args.max_runtime_hours,
                code=git_state(root),
                data_artifact=data,
                model_artifact=model,
                producer_evidence=(
                    _json_value(args.producer_evidence) if args.producer_evidence else None
                ),
            )
            ledger.append(ExperimentEvent.create(run_id, EventType.REGISTERED, payload))
            print(run_id)
            return 0
        if args.experiment_command == "repair":
            print(ledger.repair_truncated(args.reason))
            return 0
        if args.experiment_command == "start":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.STARTED,
                start_payload(
                    kaggle_ref=args.kaggle_ref,
                    authorization_id=args.authorization_id,
                    quota_before_hours=args.quota_before_hours,
                ),
            )
        elif args.experiment_command == "finish":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.COMPLETED,
                completed_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    metrics=_load_metrics(args.metrics_report),
                    artifacts=_artifact_records(root, args.artifact),
                    reports=_artifact_records(root, args.report),
                    public_score=args.public_score,
                    producer_evidence=(
                        _json_value(args.producer_evidence) if args.producer_evidence else None
                    ),
                ),
            )
        elif args.experiment_command == "fail":
            traceback_artifact = artifact_record(root, args.traceback) if args.traceback else None
            event = ExperimentEvent.create(
                args.run_id,
                EventType.FAILED,
                failed_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    failure_reason=args.reason,
                    traceback_artifact=traceback_artifact,
                ),
            )
        elif args.experiment_command == "reject":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.REJECTED,
                rejected_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    failed_gate=args.gate,
                    reason=args.reason,
                ),
            )
        elif args.experiment_command == "decide":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.DECISION,
                decision_payload(args.decision, args.evidence),
            )
        elif args.experiment_command == "amend":
            replacement = _json_value(args.replacement)
            target = next(
                (
                    candidate
                    for candidate in ledger.read_events()
                    if candidate.event_id == args.target_event_id and candidate.run_id == args.run_id
                ),
                None,
            )
            if target is None:
                raise ValueError("amendment target was not found for this run")
            event = ExperimentEvent.create(
                args.run_id,
                EventType.AMENDMENT,
                amendment_payload(
                    target_event_id=args.target_event_id,
                    correction_reason=args.reason,
                    replacement_fields=replacement,
                ),
            )
            ledger.append(event)
            print(
                json.dumps(
                    {"original": target.payload, "corrected": replacement, "event_id": event.event_id},
                    sort_keys=True,
                )
            )
            return 0
        else:
            parser.error(f"unknown experiment command: {args.experiment_command}")
        ledger.append(event)
        print(event.event_id)
        return 0
    if args.command == "evaluate":
        from .evaluation import ExactEvaluationRequest, evaluate_exact

        request = ExactEvaluationRequest(
            ledger_path=_rooted(root, args.ledger),
            evaluation_run_id=args.evaluation_run_id,
            truth_dir=_rooted(root, args.truth_dir),
            manifest_path=_rooted(root, args.manifest),
            scorer_lock_path=_rooted(root, args.scorer_lock),
            evaluation_policy_path=_rooted(root, args.evaluation_policy),
            baseline_sets=_prediction_refs(root, args.baseline_set),
            candidate_sets=_prediction_refs(root, args.candidate_set),
            output_dir=_rooted(root, args.output_dir),
            scorer_checkout=_rooted(
                root,
                args.scorer_checkout
                or Path(".biohub/vendor/kaggle-cell-tracking-competition"),
            ),
            tracksdata_checkout=_rooted(
                root, args.tracksdata_checkout or Path(".biohub/vendor/tracksdata")
            ),
            workspace_root=root,
        )
        report = evaluate_exact(request)
        print(
            json.dumps(
                {
                    "report_core_sha256": report.core_sha256,
                    "envelope_sha256": report.envelope_sha256,
                    "core_path": str(report.core_path),
                    "envelope_path": str(report.envelope_path),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "promote":
        from .evaluation import ExactReport
        from .promotion import evaluate_promotion, record_promotion

        report_root = _rooted(root, args.report)
        report = ExactReport.from_files(
            report_root / "exact-report-core.json",
            report_root / "exact-report-envelope.json",
        )
        kwargs = {
            "policy_path": _rooted(root, args.policy),
            "ledger_path": _rooted(root, args.ledger),
            "workspace_root": root,
            "evaluation_run_id": args.evaluation_run_id,
        }
        if args.record:
            event = record_promotion(report, **kwargs)
            result = {"recorded_event_id": event.event_id, **dict(event.payload)}
        else:
            result = evaluate_promotion(report, **kwargs).to_dict()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "cpu-acceptance":
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        if args.cpu_acceptance_command == "build-bundle":
            from .acceptance import build_runtime_bundle

            result = build_runtime_bundle(
                _rooted(root, args.source), _rooted(root, args.output)
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        if args.cpu_acceptance_command == "verify-bundle":
            from .acceptance import (
                _source_inventory_sha256,
                extract_runtime_bundle,
                inspect_runtime_bundle,
            )

            bundle_path = _rooted(root, args.bundle)
            expected = inspect_runtime_bundle(bundle_path)
            result = extract_runtime_bundle(
                bundle_path,
                _rooted(root, args.destination),
                expected_sha256=expected["runtime_bundle_sha256"],
                expected_inventory_sha256=expected[
                    "runtime_bundle_inventory_sha256"
                ],
                expected_uncompressed_size_bytes=expected[
                    "runtime_bundle_uncompressed_size_bytes"
                ],
                expected_file_count=expected["runtime_bundle_file_count"],
            )
            extracted_root = _rooted(root, args.destination) / "bundle"
            if _source_inventory_sha256(extracted_root) != _source_inventory_sha256(
                root
            ):
                raise ValueError("runtime bundle source identity differs from workspace")
            extracted_config = _load_config(
                extracted_root / "config" / "phase2-control.json"
            )
            local_config = _load_config(root / "config" / "phase2-control.json")
            if extracted_config != local_config:
                raise ValueError("runtime bundle control config differs from workspace")
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        if args.cpu_acceptance_command == "preflight":
            from .ledger import (
                CpuAcceptanceStatus,
                ExactEvaluationStatus,
                reconstruct_exact_evaluations,
            )

            events = ledger.read_events()
            cpu_active = sorted(
                run_id
                for run_id, state in reconstruct_cpu_acceptances(events).items()
                if state.status
                not in {CpuAcceptanceStatus.COMPLETED, CpuAcceptanceStatus.FAILED}
            )
            exact_active = sorted(
                run_id
                for run_id, state in reconstruct_exact_evaluations(events).items()
                if state.status is ExactEvaluationStatus.RUNNING
            )
            if cpu_active or exact_active:
                raise ValueError(
                    f"nonterminal acceptance lifecycle: cpu={cpu_active}, exact={exact_active}"
                )
            print(json.dumps({"cpu_active": [], "exact_active": []}, sort_keys=True))
            return 0
        if args.cpu_acceptance_command == "register":
            if args.config is not None:
                from .acceptance import issue_acceptance_request

                required = {
                    "--run-id": args.run_id,
                    "--evaluation-run-id": args.evaluation_run_id,
                    "--kernel-ref": args.kernel_ref,
                    "--runtime-dataset-ref": args.runtime_dataset_ref,
                    "--runtime-bundle": args.runtime_bundle,
                    "--output": args.output,
                }
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    raise ValueError(
                        f"config registration requires {', '.join(sorted(missing))}"
                    )
                _, event = issue_acceptance_request(
                    workspace_root=root,
                    ledger_path=ledger.path,
                    config_path=_rooted(root, args.config),
                    run_id=args.run_id,
                    evaluation_run_id=args.evaluation_run_id,
                    kernel_ref=args.kernel_ref,
                    runtime_dataset_ref=args.runtime_dataset_ref,
                    runtime_bundle_path=_rooted(root, args.runtime_bundle),
                    output_path=_rooted(root, args.output),
                )
                print(event.event_id)
                return 0
            from .acceptance import validate_registered_acceptance_request

            request = _load_config(_rooted(root, args.request))
            event = validate_registered_acceptance_request(
                request, ledger.read_events()
            )
        elif args.cpu_acceptance_command == "start":
            states = reconstruct_cpu_acceptances(ledger.read_events())
            state = states.get(args.run_id)
            if state is None:
                raise ValueError(f"unknown CPU acceptance ID: {args.run_id}")
            registration_event = next(
                item
                for item in state.events
                if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
            )
            event = ExperimentEvent.create(
                args.run_id,
                EventType.CPU_ACCEPTANCE_STARTED,
                cpu_acceptance_started_payload(
                    run_id=args.run_id,
                    registration_event_sha256=event_sha256(registration_event),
                    kernel_ref=args.kernel_ref,
                    runtime_dataset_ref=args.runtime_dataset_ref,
                ),
            )
            ledger.append(event)
        elif args.cpu_acceptance_command == "reconcile":
            from .acceptance import reconcile_pending_control

            result = reconcile_pending_control(
                pending_path=_rooted(root, args.evidence),
                ledger_path=ledger.path,
                workspace_root=root,
                manifest_output_path=_rooted(root, args.manifest_output),
                report_output_path=_rooted(root, args.report_output),
                quota_before_path=_rooted(root, args.quota_before),
                quota_after_path=_rooted(root, args.quota_after),
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        else:
            from .ledger import CpuAcceptanceStatus, ExactEvaluationStatus

            appended = []
            state = reconstruct_cpu_acceptances(ledger.read_events()).get(args.run_id)
            if state is None:
                raise ValueError(f"unknown CPU acceptance ID: {args.run_id}")
            if state.status not in {
                CpuAcceptanceStatus.COMPLETED,
                CpuAcceptanceStatus.FAILED,
            }:
                event = ExperimentEvent.create(
                    args.run_id,
                    EventType.CPU_ACCEPTANCE_FAILED,
                    cpu_acceptance_failed_payload(
                        run_id=args.run_id,
                        reason_code=args.reason_code,
                        detail=args.detail,
                        observed_artifact_hashes=(
                            _json_value(args.observed_artifacts)
                            if args.observed_artifacts
                            else None
                        ),
                    ),
                )
                ledger.append(event)
                appended.append(event.event_id)
            if args.evaluation_run_id:
                from .ledger import (
                    exact_evaluation_failed_payload,
                    reconstruct_exact_evaluations,
                )

                evaluation = reconstruct_exact_evaluations(ledger.read_events()).get(
                    args.evaluation_run_id
                )
                if evaluation and evaluation.status is ExactEvaluationStatus.RUNNING:
                    exact_failure = ExperimentEvent.create(
                        args.evaluation_run_id,
                        EventType.EXACT_EVALUATION_FAILED,
                        exact_evaluation_failed_payload(
                            evaluation_run_id=args.evaluation_run_id,
                            reason_code=args.reason_code,
                            detail=args.detail,
                        ),
                    )
                    ledger.append(exact_failure)
                    appended.append(exact_failure.event_id)
            print(json.dumps({"failed_events": appended}, sort_keys=True))
            return 0
        print(event.event_id)
        return 0
    if args.command == "progress":
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        events = ledger.read_events()
        markdown_path, _ = write_progress_reports(events, root)
        if args.json_output:
            print(json.dumps(render_progress_json(events), indent=2, sort_keys=True))
        else:
            print(render_progress_markdown(events))
            print(f"\nReport: {markdown_path}")
        return 0
    if args.command == "guard":
        config = _load_config(root / "config" / "competition.json")
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        runs = reconstruct_runs(ledger.read_events())
        state = runs.get(args.run_id)
        fixture_dir = args.fixture_dir or (root / "tests" / "fixtures" / "kaggle")
        runner = KaggleRunner() if args.live else FixtureRunner(fixture_dir)
        quota = None
        active = []
        input_error = None
        try:
            active = list_active_gpu_kernels(runner, config["slug"])
            quota = read_gpu_quota(runner)
        except GuardInputError as exc:
            input_error = exc.reason_code
        decision = evaluate_guard(
            run_id=args.run_id,
            registered_status=state.status if state else None,
            registered_declared_runtime=(
                state.registered.get("declared_max_runtime_hours") if state else None
            ),
            declared_max_runtime=args.max_runtime_hours,
            quota=quota,
            active_kernels=active,
            reserve=config["gpu_reserve_hours"],
            notebook_maximum=config["notebook_runtime_limit_hours"],
            input_error_code=input_error,
        )
        if state:
            ledger.append(
                ExperimentEvent.create(args.run_id, EventType.GUARD_DECISION, decision.to_dict())
            )
        print(json.dumps(decision.to_dict(), indent=2 if args.json_output else None, sort_keys=True))
        return 0 if decision.authorized else 2
    if args.command == "preflight":
        report = validate_preflight(
            args.report,
            root,
            run_id=args.run_id,
            declared_max_runtime_hours=args.max_runtime_hours,
            max_age_seconds=args.max_age_seconds,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "launch":
        config = _load_config(root / "config" / "competition.json")
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        fixture_dir = args.fixture_dir or (root / "tests" / "fixtures" / "kaggle")
        runner = KaggleRunner() if args.live else FixtureRunner(fixture_dir)
        if args.launch_command == "authorize":
            authorization, decision = authorize_launch(
                workspace_root=root,
                ledger=ledger,
                runner=runner,
                config=config,
                run_id=args.run_id,
                kernel_directory=args.kernel_dir,
                kernel_ref=args.kernel_ref,
                preflight_report=args.preflight_report,
                ttl_seconds=args.ttl_seconds,
            )
            print(
                json.dumps(
                    {"authorization": authorization.to_dict(), "guard": decision.to_dict()},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        authorization = load_authorization(root, args.authorization_id)
        if not args.execute:
            print(
                json.dumps(
                    {"executed": False, "authorization": authorization.to_dict()},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not args.nonce:
            raise ValueError("--nonce is required with --execute")
        if not args.live and launch_runner is None:
            raise ValueError(
                "actual launch requires --live; fixtures are only accepted with an injected test runner"
            )
        result = push_kernel(
            authorization,
            workspace_root=root,
            ledger=ledger,
            runner=runner,
            config=config,
            nonce=args.nonce,
            execute=True,
            push_runner=launch_runner or default_push_runner,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "scorer":
        from .scorer import run_fixture_tracer
        from .scorer_lock import corroborate_live, verify_scorer_lock

        lock_path = args.lock or root / "config" / "official-scorer.lock.json"
        checkout = args.checkout or root / ".biohub" / "vendor" / "kaggle-cell-tracking-competition"
        tracksdata_checkout = args.tracksdata_checkout or root / ".biohub" / "vendor" / "tracksdata"
        verified = verify_scorer_lock(
            lock_path,
            checkout,
            tracksdata_checkout=tracksdata_checkout,
        )
        result = run_fixture_tracer(
            verified,
            args.fixture
            or root / "tests" / "fixtures" / "metric" / "graph_specs" / "perfect-linear.json",
            args.expected
            or root / "tests" / "fixtures" / "metric" / "expected" / "official-counts.json",
        )
        if args.live:
            result["live_provenance"] = corroborate_live(verified.lock)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "manifest":
        from .manifests import build_manifest, verify_manifest, write_manifest

        scorer_lock = args.scorer_lock or root / "config" / "official-scorer.lock.json"
        if args.manifest_command == "build":
            manifest = build_manifest(args.data_root, scorer_lock)
            path = write_manifest(args.output, manifest)
            print(json.dumps({"manifest": str(path), "manifest_sha256": manifest.manifest_sha256}, sort_keys=True))
            return 0
        manifest = verify_manifest(
            args.manifest,
            data_root=args.data_root,
            scorer_lock_path=scorer_lock if args.data_root is not None else None,
        )
        print(json.dumps({"manifest_sha256": manifest.manifest_sha256, "verified": True}, sort_keys=True))
        return 0
    if args.command == "graph":
        from dataclasses import asdict

        from .graphs import load_geff_graph, preflight_prediction_set, validate_prediction_inventory
        from .scorer_lock import verify_scorer_lock

        ledger_path = args.ledger if args.ledger.is_absolute() else root / args.ledger
        inventory = preflight_prediction_set(
            args.pred_dir,
            args.producer_manifest,
            args.manifest,
            args.fold,
            Ledger(ledger_path, root),
        )
        verified = verify_scorer_lock(
            args.scorer_lock or root / "config" / "official-scorer.lock.json",
            args.scorer_checkout
            or root / ".biohub" / "vendor" / "kaggle-cell-tracking-competition",
            tracksdata_checkout=args.tracksdata_checkout
            or root / ".biohub" / "vendor" / "tracksdata",
        )
        result = validate_prediction_inventory(
            inventory,
            mode=args.mode,
            graph_loader=lambda path: load_geff_graph(path, verified),
        )
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "submission":
        from .graphs import preflight_prediction_set
        from .scorer_lock import verify_scorer_lock
        from .submission_io import roundtrip_prediction_inventory

        ledger_path = args.ledger if args.ledger.is_absolute() else root / args.ledger
        inventory = preflight_prediction_set(
            args.pred_dir,
            args.producer_manifest,
            args.manifest,
            args.fold,
            Ledger(ledger_path, root),
        )
        verified = verify_scorer_lock(
            args.scorer_lock or root / "config" / "official-scorer.lock.json",
            args.scorer_checkout
            or root / ".biohub" / "vendor" / "kaggle-cell-tracking-competition",
            tracksdata_checkout=args.tracksdata_checkout
            or root / ".biohub" / "vendor" / "tracksdata",
        )
        evidence = roundtrip_prediction_inventory(
            inventory, verified, args.truth_root, args.output_dir
        )
        print(json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None, *, launch_runner=None) -> int:
    try:
        return _main(argv, launch_runner=launch_runner)
    except (GuardInputError, LedgerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
