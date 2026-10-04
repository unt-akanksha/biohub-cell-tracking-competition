from __future__ import annotations

import base64
import inspect
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_tracker.acceptance import (
    AcceptanceError,
    BUNDLE_NAME,
    PendingControlReport,
    _assert_control_projection_matches,
    _append_events_then_publish_acceptance,
    _assert_trusted_control_matches,
    _local_control_trust,
    _preflight_immutable_json,
    _reuse_saga_event,
    _saga_event,
    _validate_truth_self_sufficient_row,
    _bundle_member_path,
    assert_cpu_kernel_metadata,
    assert_no_submission_source,
    build_runtime_bundle,
    extract_runtime_bundle,
    issue_acceptance_request,
    validate_pending_control,
)
from biohub_tracker.cli import main
from biohub_tracker.evaluation import (
    ExactEvaluationRequest,
    evaluate_pending_control,
    validate_exact_report,
)
from biohub_tracker.io import canonical_json_bytes, sha256_bytes, sha256_file
from biohub_tracker.graphs import artifact_tree_sha256
from biohub_tracker.ledger import (
    EventType,
    ExperimentEvent,
    cpu_acceptance_registration_payload,
    event_sha256,
)


def _pending() -> dict:
    payload = {
        "schema_version": "biohub.pending-control-report.v1",
        "status": "pending_reconciliation",
        "run_id": "cpu-control-a",
        "evaluation_run_id": "eval-control-a",
        "request_nonce": "1" * 64,
        "acceptance_request_sha256": "2" * 64,
        "registration_event_sha256": "3" * 64,
        "kernel_ref": "indarkarhana/biohub-phase-2-cpu-acceptance/1",
        "runtime_dataset_ref": "indarkarhana/biohub-phase2-runtime/1",
        "runtime_bundle_name": BUNDLE_NAME,
        "runtime_bundle_sha256": "b" * 64,
        "runtime_bundle_inventory_sha256": "c" * 64,
        "runtime_bundle_uncompressed_size_bytes": 1000,
        "runtime_bundle_file_count": 3,
        "accelerator": "none",
        "internet_enabled": False,
        "competition_submission_performed": False,
        "watchdog_terminal_state": "completed",
        "actual_cpu_runtime_seconds": "10",
        "peak_memory_mb": "100",
        "source_identities": {
            "scorer_lock_sha256": "4" * 64,
            "environment_lock_sha256": "5" * 64,
            "manifest_policy_sha256": "6" * 64,
            "control_model_sha256": "7" * 64,
            "config_sha256": "8" * 64,
            "code_sha256": "9" * 64,
            "data_source_sha256": "a" * 64,
        },
        "manifest": {
            "manifest_sha256": "b" * 64,
            "folds": [
                {
                    "fold_id": "fold-44b6-to-6bba",
                    "train_membership_sha256": "c" * 64,
                    "calibration_membership_sha256": "d" * 64,
                    "evaluation_membership_sha256": "e" * 64,
                },
                {
                    "fold_id": "fold-6bba-to-44b6",
                    "train_membership_sha256": "f" * 64,
                    "calibration_membership_sha256": "0" * 64,
                    "evaluation_membership_sha256": "1" * 64,
                },
            ],
            "sample_count": 2,
            "overlap_count": 0,
            "manifest_document": {"schema_version": 1},
        },
        "control": {
            "graph_inventory_sha256": "2" * 64,
            "artifact_hashes": {"truth_graphs": "3" * 64},
            "authoritative_inventories": [],
            "expected_sample_ids": ["44b6_a", "6bba_a"],
            "official": {},
            "diagnostics": {},
            "comparison": {},
            "evaluation_policy_sha256": "4" * 64,
            "provisional_members": [],
        },
    }
    payload["pending_payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    envelope = {
        "schema_version": "biohub.pending-control-envelope.v1",
        "pending_payload_sha256": payload["pending_payload_sha256"],
        "authority": "remote_untrusted_pending",
    }
    payload["pending_envelope"] = envelope
    payload["pending_envelope_sha256"] = sha256_bytes(canonical_json_bytes(envelope))
    semantic = dict(payload)
    semantic.pop("output_inventory_sha256", None)
    payload["output_inventory_sha256"] = sha256_bytes(canonical_json_bytes(semantic))
    return payload


def test_pending_control_is_strict_untrusted_and_not_an_exact_report():
    value = _pending()
    pending = validate_pending_control(value)
    assert isinstance(pending, PendingControlReport)
    assert pending.accelerator == "none"
    with pytest.raises(Exception):
        validate_exact_report(pending)  # type: ignore[arg-type]
    tampered = dict(value)
    tampered["request_nonce"] = "f" * 64
    with pytest.raises(AcceptanceError, match="PENDING_PAYLOAD_HASH_MISMATCH"):
        validate_pending_control(tampered)


def test_reconciliation_saga_events_and_artifacts_are_retry_stable(tmp_path: Path):
    payload = {"run_id": "cpu-control-a", "manifest_sha256": "a" * 64, "folds": []}
    first = _saga_event(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
        payload,
        created_at="2026-08-25T00:00:00Z",
    )
    retry = _saga_event(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
        payload,
        created_at="2026-08-25T00:00:00Z",
    )
    assert first == retry
    assert _reuse_saga_event([first], retry) == (first, False)

    target = tmp_path / "accepted.json"
    assert _preflight_immutable_json(target, {"accepted": True}) is True
    target.write_text('{"accepted": true}\n', encoding="utf-8")
    assert _preflight_immutable_json(target, {"accepted": True}) is False
    with pytest.raises(AcceptanceError, match="IMMUTABLE_ACCEPTANCE_OUTPUT_CONFLICT"):
        _preflight_immutable_json(target, {"accepted": False})

    conflicting = _saga_event(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
        {**payload, "manifest_sha256": "b" * 64},
        created_at="2026-08-25T00:00:00Z",
    )
    with pytest.raises(AcceptanceError, match="RECONCILIATION_EVENT_CONFLICT"):
        _reuse_saga_event([first], conflicting)


def test_acceptance_is_not_published_before_all_terminal_events(monkeypatch, tmp_path):
    import biohub_tracker.acceptance as acceptance_module

    appended = []
    writes = []
    events = tuple(object() for _ in range(3))

    def interrupted(_ledger, event):
        appended.append(event)
        if len(appended) == 2:
            raise SystemExit("kill during ledger commit")

    monkeypatch.setattr(acceptance_module, "_append_saga_event", interrupted)
    monkeypatch.setattr(
        acceptance_module,
        "atomic_write_json",
        lambda path, value: writes.append((path, value)),
    )
    with pytest.raises(SystemExit, match="ledger commit"):
        _append_events_then_publish_acceptance(
            ledger=object(),
            events=events,
            report_target=tmp_path / "accepted.json",
            accepted={"accepted": True},
            write_report=True,
        )
    assert writes == []

    appended.clear()
    monkeypatch.setattr(
        acceptance_module,
        "_append_saga_event",
        lambda _ledger, event: appended.append(event),
    )
    _append_events_then_publish_acceptance(
        ledger=object(),
        events=events,
        report_target=tmp_path / "accepted.json",
        accepted={"accepted": True},
        write_report=True,
    )
    assert appended == list(events)
    assert writes == [(tmp_path / "accepted.json", {"accepted": True})]


@pytest.mark.parametrize("section", ["official", "diagnostics", "comparison"])
def test_remote_self_hashes_do_not_authorize_tampered_control_projection(section: str):
    control = {
        "official": {"pooled": {"edge_tp": 10}},
        "diagnostics": {"pooled": {"endpoint_available": 10}},
        "comparison": {"pooled": {"score_delta": "0"}},
    }
    trusted = json.loads(json.dumps(control))
    control[section]["tampered"] = True

    # A remote producer can recompute its own payload/envelope hashes. The local
    # semantic recomputation is an independent equality boundary and still fails.
    with pytest.raises(AcceptanceError, match=f"CONTROL_REVALIDATION_MISMATCH: {section}"):
        _assert_control_projection_matches(
            control,
            official=trusted["official"],
            diagnostics=trusted["diagnostics"],
            comparison=trusted["comparison"],
        )


def test_truth_self_count_tamper_is_rejected_even_when_redundant_fields_match():
    counts = {
        "edge_tp": 4,
        "edge_fp": 0,
        "edge_fn": 0,
        "division_tp": 1,
        "division_fp": 0,
        "division_fn": 0,
        "num_pred_nodes": 5,
    }
    row = {
        "sample_id": "embryo_movie",
        "official_counts": dict(counts),
        "organizer_row": dict(counts),
        "matched_gt_node_count": 5,
        "node_recall": "1",
        "diagnostic_state": {
            "reconciliation": {
                "status": "passed",
                **{name: value for name, value in counts.items() if name != "num_pred_nodes"},
            }
        },
    }
    sample = SimpleNamespace(gt_edge_count=4, gt_node_count=5)
    trusted = json.loads(json.dumps(row))
    _validate_truth_self_sufficient_row(row, sample, trusted_row=trusted)

    # A producer can change every redundant remote count and all self-hashes,
    # but it cannot change the locally reconstructed manifest edge total.
    row["official_counts"]["edge_tp"] = 3
    row["organizer_row"]["edge_tp"] = 3
    row["diagnostic_state"]["reconciliation"]["edge_tp"] = 3
    with pytest.raises(AcceptanceError, match="CONTROL_TRUST_BINDING_MISMATCH"):
        _validate_truth_self_sufficient_row(row, sample, trusted_row=trusted)


@pytest.mark.parametrize(
    "target",
    [
        "division_tp",
        "score",
        "diagnostic",
        "submission_graph_inventory_sha256",
        "roundtrip_evidence_sha256",
        "csv_sha256",
    ],
)
def test_remote_rehash_cannot_change_locally_content_bound_control_evidence(target):
    trusted_rows = [
        {
            "sample_id": "44b6_a",
            "official_counts": {"division_tp": 2},
            "organizer_row": {"score": "1.1"},
            "diagnostic_state": {"density": {"observed": 3}},
        }
    ]
    trusted_authoritative = [
        {
            "role": "candidate",
            "fold_id": "fold-a",
            "producer_run_id": "cpu-control-a",
            "submission_graph_inventory_sha256": "1" * 64,
            "roundtrip_evidence_sha256": "2" * 64,
            "csv_sha256": "3" * 64,
        }
    ]
    remote_rows = json.loads(json.dumps(trusted_rows))
    remote_authoritative = json.loads(json.dumps(trusted_authoritative))
    if target == "division_tp":
        remote_rows[0]["official_counts"]["division_tp"] = 999
    elif target == "score":
        remote_rows[0]["organizer_row"]["score"] = "999"
    elif target == "diagnostic":
        remote_rows[0]["diagnostic_state"]["density"]["observed"] = 999
    else:
        remote_authoritative[0][target] = "f" * 64

    # A remote producer may recompute all pending envelope/projection hashes;
    # these values still cannot differ from the independently staged artifacts.
    with pytest.raises(AcceptanceError, match="CONTROL_(TRUST|INVENTORY)_BINDING_MISMATCH"):
        _assert_trusted_control_matches(
            remote_rows=remote_rows,
            trusted_rows=trusted_rows,
            remote_authoritative=remote_authoritative,
            trusted_authoritative=trusted_authoritative,
        )


def test_local_control_trust_verifies_actual_csv_geff_and_evidence_bytes(tmp_path):
    trust_root = tmp_path / "manifest-control-trust"
    artifact_root = trust_root / "fold-a"
    graph_root = artifact_root / "graphs" / "44b6_a.geff"
    graph_root.mkdir(parents=True)
    (graph_root / "zarr.json").write_text("{}", encoding="utf-8")
    csv_path = artifact_root / "submission.csv"
    csv_path.write_text("id,dataset\n", encoding="utf-8")
    graph_records = [
        {
            "sample_id": "44b6_a",
            "path": "graphs/44b6_a.geff",
            "sha256": artifact_tree_sha256(graph_root),
            "node_count": 1,
            "edge_count": 0,
        }
    ]
    evidence = {
        "producer_run_id": "trusted-builder",
        "fold_id": "fold-a",
        "manifest_sha256": "a" * 64,
        "source_graph_inventory_sha256": "b" * 64,
        "submission_graph_inventory_sha256": sha256_bytes(
            canonical_json_bytes(graph_records)
        ),
        "csv_sha256": sha256_file(csv_path),
        "source_space": "native-geff",
        "authoritative_space": "integer-csv-rebuilt-geff",
        "semantic_parity": True,
        "official_counts_parity": True,
        "source_lineage": {},
        "graphs": graph_records,
        "per_movie": [],
        "aggregate": {},
        "schema_version": 1,
    }
    evidence["evidence_sha256"] = sha256_bytes(canonical_json_bytes(evidence))
    (artifact_root / "roundtrip-evidence.json").write_bytes(
        canonical_json_bytes(evidence) + b"\n"
    )
    rows_path = trust_root / "movie-rows.json"
    rows_path.write_text("[]", encoding="utf-8")
    semantic_trust = {
        "schema_version": "biohub.local-control-trust.v1",
        "manifest_sha256": "a" * 64,
        "movie_rows_relpath": "movie-rows.json",
        "movie_rows_sha256": sha256_file(rows_path),
        "fold_artifacts": [
            {
                "fold_id": "fold-a",
                "artifact_relpath": "fold-a",
                "artifact_tree_sha256": artifact_tree_sha256(artifact_root),
            }
        ],
    }
    trust = dict(semantic_trust)
    trust["control_trust_sha256"] = sha256_bytes(canonical_json_bytes(semantic_trust))
    (trust_root / "control-trust.json").write_bytes(canonical_json_bytes(trust) + b"\n")
    manifest = SimpleNamespace(
        manifest_sha256="a" * 64, folds=(SimpleNamespace(fold_id="fold-a"),)
    )
    pending = SimpleNamespace(run_id="cpu-control-a")

    rows, inventories = _local_control_trust(trust_root, manifest, pending)
    assert rows == []
    assert len(inventories) == 2
    csv_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="LOCAL_CONTROL_TRUST_INVALID"):
        _local_control_trust(trust_root, manifest, pending)


@pytest.mark.parametrize(
    "field,value",
    [("enable_gpu", True), ("enable_tpu", True), ("enable_internet", True)],
)
def test_cpu_kernel_metadata_fail_closed_on_accelerators_or_internet(field, value):
    metadata = {
        "id": "indarkarhana/biohub-phase-2-cpu-acceptance",
        "code_file": "phase2_acceptance.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["indarkarhana/biohub-phase2-runtime"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
    }
    metadata[field] = value
    with pytest.raises(AcceptanceError):
        assert_cpu_kernel_metadata(
            metadata,
            expected_kernel_slug="indarkarhana/biohub-phase-2-cpu-acceptance",
            expected_dataset_slug="indarkarhana/biohub-phase2-runtime",
            expected_competition_slug="biohub-cell-tracking-during-development",
        )


def test_submission_and_gpu_launch_tokens_are_unreachable():
    safe = "kaggle kernels push -p owned_cpu_kernel\n# accelerator none"
    assert_no_submission_source(safe)
    for forbidden in (
        "kaggle competitions submit -c biohub",
        "torch.cuda.is_available()",
        "biohub launch execute --execute",
    ):
        with pytest.raises(AcceptanceError):
            assert_no_submission_source(forbidden)


def test_exact_cli_builds_four_role_fold_refs_and_requires_ledger(tmp_path, monkeypatch):
    captured: dict[str, ExactEvaluationRequest] = {}

    def fake_evaluate(request):
        captured["request"] = request
        return type(
            "Report",
            (),
            {
                "core_sha256": "1" * 64,
                "envelope_sha256": "2" * 64,
                "core_path": tmp_path / "core.json",
                "envelope_path": tmp_path / "envelope.json",
            },
        )()

    monkeypatch.setattr("biohub_tracker.evaluation.evaluate_exact", fake_evaluate)
    args = [
        "--root",
        str(tmp_path),
        "evaluate",
        "exact",
        "--ledger",
        "experiments/events.jsonl",
        "--evaluation-run-id",
        "eval-a",
        "--truth-dir",
        "truth",
        "--manifest",
        "manifest.json",
        "--scorer-lock",
        "lock.json",
        "--evaluation-policy",
        "policy.json",
        "--output-dir",
        "out",
    ]
    for role in ("baseline", "candidate"):
        for fold in ("fold-a", "fold-b"):
            args += [f"--{role}-set", f"{fold}={role}-{fold}"]
    assert main(args) == 0
    request = captured["request"]
    assert request.ledger_path == (tmp_path / "experiments/events.jsonl").resolve()
    assert [item.fold_id for item in request.baseline_sets] == ["fold-a", "fold-b"]
    assert [item.fold_id for item in request.candidate_sets] == ["fold-a", "fold-b"]


def test_tracked_kernel_metadata_and_source_have_cpu_tripwires():
    metadata = json.loads(
        Path("kaggle/phase2-cpu-acceptance/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert_cpu_kernel_metadata(
        metadata,
        expected_kernel_slug="indarkarhana/biohub-phase-2-cpu-acceptance",
        expected_dataset_slug="indarkarhana/biohub-phase2-runtime",
        expected_competition_slug="biohub-cell-tracking-during-development",
    )
    source = Path("kaggle/phase2-cpu-acceptance/phase2_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert_no_submission_source(source)
    assert 'runtime / "control-work", package_root' in source
    assert "generated_output_cleanup_failed" in source


def test_pending_control_uses_the_validated_evaluation_policy_loader():
    source = inspect.getsource(evaluate_pending_control)
    assert "load_evaluation_policy(evaluation_policy_path)" in source
    assert "_load_policy(" not in source


def test_kernel_only_retry_embeds_a_fresh_request_without_changing_runtime_dataset():
    source = Path("kaggle/phase2-cpu-acceptance/phase2_acceptance.py").read_text(
        encoding="utf-8"
    )
    request = {"schema_version": "biohub.acceptance-request.v1", "run_id": "fresh"}
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    staged = source.replace(
        'EMBEDDED_ACCEPTANCE_REQUEST_B64 = ""',
        f'EMBEDDED_ACCEPTANCE_REQUEST_B64 = "{encoded}"',
    )
    assert staged != source
    namespace: dict = {}
    exec(compile(staged, "phase2_acceptance.py", "exec"), namespace)
    runtime = Path("unused-because-the-request-is-embedded")
    assert namespace["_acceptance_request"](runtime) == request


def _kernel_script_namespace() -> dict:
    return runpy.run_path("kaggle/phase2-cpu-acceptance/phase2_acceptance.py")


def _kernel_state_namespace() -> dict:
    return runpy.run_path("scripts/get-kaggle-kernel-state.py")


def _dataset_state_namespace() -> dict:
    return runpy.run_path("scripts/get-kaggle-dataset-state.py")


def test_mount_discovery_accepts_direct_source(tmp_path):
    marker = "bundle/config/official-scorer.lock.json"
    source = tmp_path / "biohub-phase2-runtime"
    (source / marker).parent.mkdir(parents=True)
    (source / marker).write_text("{}", encoding="utf-8")
    assert _kernel_script_namespace()["_one_with_marker"](marker, tmp_path) == source


def test_mount_discovery_accepts_owner_qualified_source(tmp_path):
    marker = "bundle/config/official-scorer.lock.json"
    source = tmp_path / "indarkarhana" / "biohub-phase2-runtime"
    (source / marker).parent.mkdir(parents=True)
    (source / marker).write_text("{}", encoding="utf-8")
    assert _kernel_script_namespace()["_one_with_marker"](marker, tmp_path) == source


def test_mount_discovery_rejects_missing_source_with_safe_diagnostic(tmp_path):
    marker = "bundle/config/official-scorer.lock.json"
    with pytest.raises(RuntimeError) as caught:
        _kernel_script_namespace()["_one_with_marker"](marker, tmp_path)
    detail = str(caught.value)
    assert detail == (
        "mounted_source_cardinality: marker=official-scorer.lock.json "
        "direct=0 owner_qualified=0 namespaced=0"
    )
    assert str(tmp_path) not in detail


@pytest.mark.parametrize("owner_qualified", [False, True])
def test_opaque_bundle_discovery_accepts_bounded_kaggle_layouts(
    tmp_path, owner_qualified
):
    source = tmp_path / "biohub-phase2-runtime"
    if owner_qualified:
        source = tmp_path / "indarkarhana" / "biohub-phase2-runtime"
    source.mkdir(parents=True)
    bundle = source / BUNDLE_NAME
    bundle.write_bytes(b"opaque")
    assert _kernel_script_namespace()["_one_bundle"](BUNDLE_NAME, tmp_path) == bundle


def test_current_kaggle_namespaced_mount_layouts_are_bounded(tmp_path):
    namespace = _kernel_script_namespace()
    bundle = (
        tmp_path
        / "datasets"
        / "indarkarhana"
        / "biohub-phase2-runtime"
        / BUNDLE_NAME
    )
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"opaque")
    assert namespace["_one_bundle"](
        BUNDLE_NAME,
        tmp_path,
        dataset_slug="indarkarhana/biohub-phase2-runtime",
    ) == bundle

    competition = tmp_path / "competitions" / "biohub-competition"
    (competition / "train").mkdir(parents=True)
    assert namespace["_one_with_marker"](
        "train", tmp_path, competition_slug="biohub-competition"
    ) == competition


def test_opaque_bundle_discovery_rejects_zero_or_multiple(tmp_path):
    discover = _kernel_script_namespace()["_one_bundle"]
    with pytest.raises(RuntimeError, match="runtime_bundle_cardinality"):
        discover(BUNDLE_NAME, tmp_path)
    for name in ("first", "second"):
        source = tmp_path / name
        source.mkdir()
        (source / BUNDLE_NAME).write_bytes(b"opaque")
    with pytest.raises(RuntimeError, match="direct=2"):
        discover(BUNDLE_NAME, tmp_path)


def test_runtime_bundle_is_deterministic_and_securely_extracts(tmp_path):
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "second-source"
    for source in (first_source, second_source):
        (source / "bundle" / "config").mkdir(parents=True)
        (source / "bundle" / "src" / "biohub_tracker").mkdir(parents=True)
        (source / "bundle" / "config" / "lock.json").write_text(
            '{"pinned":true}\n', encoding="utf-8"
        )
        (source / "bundle" / "src" / "biohub_tracker" / "a.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
    first_output = tmp_path / "one" / BUNDLE_NAME
    second_output = tmp_path / "two" / BUNDLE_NAME
    first = build_runtime_bundle(first_source, first_output)
    second = build_runtime_bundle(second_source, second_output)
    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()
    destination = tmp_path / "extracted"
    actual = extract_runtime_bundle(
        first_output,
        destination,
        expected_sha256=first["runtime_bundle_sha256"],
        expected_inventory_sha256=first["runtime_bundle_inventory_sha256"],
        expected_uncompressed_size_bytes=first[
            "runtime_bundle_uncompressed_size_bytes"
        ],
        expected_file_count=first["runtime_bundle_file_count"],
    )
    assert actual == first
    assert (destination / "bundle" / "config" / "lock.json").read_text(
        encoding="utf-8"
    ) == '{"pinned":true}\n'


def test_runtime_bundle_rejects_hash_mismatch_and_unsafe_paths(tmp_path):
    source = tmp_path / "source"
    (source / "bundle" / "config").mkdir(parents=True)
    (source / "bundle" / "config" / "lock.json").write_text(
        "{}", encoding="utf-8"
    )
    output = tmp_path / BUNDLE_NAME
    metadata = build_runtime_bundle(source, output)
    with pytest.raises(AcceptanceError, match="RUNTIME_BUNDLE_HASH_MISMATCH"):
        extract_runtime_bundle(
            output,
            tmp_path / "bad-extract",
            expected_sha256="0" * 64,
            expected_inventory_sha256=metadata["runtime_bundle_inventory_sha256"],
            expected_uncompressed_size_bytes=metadata[
                "runtime_bundle_uncompressed_size_bytes"
            ],
            expected_file_count=metadata["runtime_bundle_file_count"],
        )
    for path in ("/bundle/config/a", "bundle/../config/a", "bundle/secrets/a"):
        with pytest.raises(AcceptanceError, match="RUNTIME_BUNDLE_PATH_INVALID"):
            _bundle_member_path(path)
    assert _bundle_member_path("bundle/tests/fixtures/metric.json").as_posix() == (
        "bundle/tests/fixtures/metric.json"
    )


def test_issue_acceptance_request_round_trips_through_cli_request_branch(
    tmp_path, monkeypatch, capsys
):
    import biohub_tracker.acceptance as acceptance_module

    config = {
        "schema_version": "biohub.phase2-control-config.v1",
        "purpose": "phase2_official_data_control",
        "competition_slug": "biohub-cell-tracking-during-development",
        "kernel_slug": "owner/biohub-phase2-cpu-acceptance",
        "runtime_dataset_slug": "owner/biohub-phase2-runtime",
        "scorer_lock_path": "config/scorer.json",
        "environment_lock_path": "config/environment.txt",
        "evaluation_policy_path": "config/evaluation.json",
        "manifest_policy": {"mode": "reciprocal"},
        "control_model": {"mode": "truth_self"},
        "cpu_watchdog_minutes": 660,
        "accelerator": "none",
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_submission_allowed": False,
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "control.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (config_dir / "scorer.json").write_text('{"lock":true}', encoding="utf-8")
    (config_dir / "environment.txt").write_text("pinned\n", encoding="utf-8")
    (config_dir / "evaluation.json").write_text('{"policy":true}', encoding="utf-8")
    bundle = tmp_path / BUNDLE_NAME
    bundle.write_bytes(b"opaque")
    monkeypatch.setattr(acceptance_module, "_source_inventory_sha256", lambda _root: "7" * 64)
    monkeypatch.setattr(
        acceptance_module,
        "inspect_runtime_bundle",
        lambda _path: {
            "runtime_bundle_name": BUNDLE_NAME,
            "runtime_bundle_sha256": "8" * 64,
            "runtime_bundle_inventory_sha256": "9" * 64,
            "runtime_bundle_uncompressed_size_bytes": 100,
            "runtime_bundle_file_count": 3,
        },
    )
    request_path = tmp_path / "request.json"
    request, event = issue_acceptance_request(
        workspace_root=tmp_path,
        ledger_path=tmp_path / "experiments" / "events.jsonl",
        config_path=config_path,
        run_id="cpu-control-roundtrip",
        evaluation_run_id="eval-control-roundtrip",
        kernel_ref="owner/biohub-phase2-cpu-acceptance/1",
        runtime_dataset_ref="owner/biohub-phase2-runtime/1",
        runtime_bundle_path=bundle,
        output_path=request_path,
    )
    before = (tmp_path / "experiments" / "events.jsonl").read_bytes()
    assert main(
        [
            "--root",
            str(tmp_path),
            "cpu-acceptance",
            "register",
            "--request",
            str(request_path),
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == event.event_id
    assert (tmp_path / "experiments" / "events.jsonl").read_bytes() == before
    assert request["source_identities"]["scorer_lock_sha256"]


@pytest.mark.parametrize("tamper", ["request_hash", "source_identity", "event_hash"])
def test_cli_request_branch_fails_closed_on_request_tampering(tamper):
    from biohub_tracker.acceptance import validate_registered_acceptance_request

    request = {
        "schema_version": "biohub.acceptance-request.v1",
        "run_id": "cpu-a",
        "evaluation_run_id": "eval-a",
        "purpose": "phase2_official_data_control",
        "request_nonce": "1" * 64,
        "competition_slug": "competition",
        "kernel_slug": "owner/kernel",
        "runtime_dataset_slug": "owner/dataset",
        "kernel_ref": "owner/kernel/1",
        "runtime_dataset_ref": "owner/dataset/1",
        "runtime_bundle_name": BUNDLE_NAME,
        "runtime_bundle_sha256": "2" * 64,
        "runtime_bundle_inventory_sha256": "3" * 64,
        "runtime_bundle_uncompressed_size_bytes": 100,
        "runtime_bundle_file_count": 3,
        "cpu_watchdog_minutes": 660,
        "accelerator": "none",
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_submission_allowed": False,
        "source_identities": {
            "scorer_lock_sha256": "4" * 64,
            "environment_lock_sha256": "5" * 64,
            "manifest_policy_sha256": "6" * 64,
            "control_model_sha256": "7" * 64,
            "config_sha256": "8" * 64,
            "code_sha256": "9" * 64,
            "data_source_sha256": "a" * 64,
        },
    }
    semantic = dict(request)
    request["acceptance_request_sha256"] = sha256_bytes(canonical_json_bytes(semantic))
    identities = request["source_identities"]
    payload = cpu_acceptance_registration_payload(
        run_id=request["run_id"],
        purpose=request["purpose"],
        request_nonce=request["request_nonce"],
        acceptance_request_sha256=request["acceptance_request_sha256"],
        evaluation_run_id=request["evaluation_run_id"],
        kernel_slug=request["kernel_slug"],
        runtime_dataset_slug=request["runtime_dataset_slug"],
        runtime_bundle_name=request["runtime_bundle_name"],
        runtime_bundle_sha256=request["runtime_bundle_sha256"],
        runtime_bundle_inventory_sha256=request["runtime_bundle_inventory_sha256"],
        runtime_bundle_uncompressed_size_bytes=request[
            "runtime_bundle_uncompressed_size_bytes"
        ],
        runtime_bundle_file_count=request["runtime_bundle_file_count"],
        scorer_lock_sha256=identities["scorer_lock_sha256"],
        environment_lock_sha256=identities["environment_lock_sha256"],
        manifest_policy_sha256=identities["manifest_policy_sha256"],
        control_model_sha256=identities["control_model_sha256"],
        config_sha256=identities["config_sha256"],
        code_sha256=identities["code_sha256"],
        data_source_sha256=identities["data_source_sha256"],
        cpu_watchdog_minutes=request["cpu_watchdog_minutes"],
    )
    registration = ExperimentEvent.create(
        "cpu-a", EventType.CPU_ACCEPTANCE_REGISTERED, payload
    )
    request["registration_event_sha256"] = event_sha256(registration)
    if tamper == "request_hash":
        request["acceptance_request_sha256"] = "c" * 64
    elif tamper == "source_identity":
        request["source_identities"]["code_sha256"] = "d" * 64
    else:
        request["registration_event_sha256"] = "e" * 64
    with pytest.raises(AcceptanceError):
        validate_registered_acceptance_request(request, [registration])


def test_kernel_only_retry_reuses_verified_runtime_dataset_and_dynamic_sdk_versions():
    wrapper = Path("scripts/run-phase2-cpu-acceptance.ps1").read_text(
        encoding="utf-8"
    )
    assert "$minimumRuntimeDatasetVersion = 9" in wrapper
    assert "$targetKernelVersion = [int]$kernelState.next_version_number" in wrapper
    assert "$kernelRef = [string]$kernelState.next_kernel_ref" in wrapper
    assert '"datasets", "download", $datasetRef' in wrapper
    assert "scripts/inspect-runtime-bundle.py" in wrapper
    assert "fully verified immutable v9 evidence" in wrapper
    assert '"datasets", "create"' not in wrapper
    assert '"datasets", "version"' not in wrapper
    assert wrapper.count('"kernels", "push"') == 1
    assert '"-t", "30"' not in wrapper
    assert '"-t", [string]$cpuWatchdogSeconds' in wrapper
    assert "$cpuWatchdogSeconds -ge (12 * 60 * 60)" in wrapper
    assert "Invoke-KaggleReadWithRetry" in wrapper
    assert "KAGGLE_READ_RATE_LIMITED" in wrapper
    assert "Start-Sleep -Seconds 60" in wrapper


def test_dataset_inventory_paginates_and_preserves_exact_file_size():
    class File:
        def __init__(self, name, total_bytes):
            self.name = name
            self.total_bytes = total_bytes

    class Page:
        def __init__(self, files, token):
            self.dataset_files = files
            self.next_page_token = token

    class Api:
        def __init__(self):
            self.calls = []

        def dataset_list_files(self, dataset, page_token=None, page_size=20):
            self.calls.append((dataset, page_token, page_size))
            if page_token is None:
                return Page([File(BUNDLE_NAME, 123)], "next")
            return Page([], None)

    api = Api()
    result = _dataset_state_namespace()["inventory_dataset"](
        "owner/runtime/3", api=api, page_size=1000
    )
    assert result["files"] == [{"name": BUNDLE_NAME, "total_bytes": 123}]
    assert api.calls == [
        ("owner/runtime/3", None, 1000),
        ("owner/runtime/3", "next", 1000),
    ]
def test_mount_discovery_rejects_duplicate_direct_and_owner_qualified_sources(tmp_path):
    marker = "bundle/config/official-scorer.lock.json"
    sources = [
        tmp_path / "biohub-phase2-runtime",
        tmp_path / "indarkarhana" / "biohub-phase2-runtime",
    ]
    for source in sources:
        (source / marker).parent.mkdir(parents=True)
        (source / marker).write_text("{}", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match=r"direct=1 owner_qualified=1 namespaced=0$",
    ):
        _kernel_script_namespace()["_one_with_marker"](marker, tmp_path)


def test_sdk_kernel_state_uses_current_version_and_absence_without_stale_reuse():
    slug = "indarkarhana/biohub-phase-2-cpu-acceptance"
    parser = _kernel_state_namespace()["kernel_state_from_sdk_response"]
    absent = parser(slug, None)
    assert absent["current_version_number"] is None
    assert absent["next_version_number"] == 1
    assert absent["next_kernel_ref"] == f"{slug}/1"

    metadata = SimpleNamespace(
        ref=slug,
        current_version_number=2,
        is_private=True,
        enable_gpu=False,
        enable_tpu=False,
        enable_internet=False,
        language="python",
        kernel_type="script",
        dataset_data_sources=["indarkarhana/biohub-phase2-runtime"],
        competition_data_sources=["biohub-cell-tracking-during-development"],
        kernel_data_sources=[],
        model_data_sources=[],
    )
    current = parser(slug, SimpleNamespace(metadata=metadata))
    assert current["current_version_number"] == 2
    assert current["next_version_number"] == 3
    assert current["next_kernel_ref"] == f"{slug}/3"


def test_owned_kernel_exact_listing_proves_only_exact_absence():
    resolver = _kernel_state_namespace()["exact_owned_kernel_refs"]

    class Api:
        def kernels_list(self, **kwargs):
            assert kwargs == {
                "search": "target-kernel",
                "mine": True,
                "page_size": 100,
            }
            return [
                SimpleNamespace(ref="indarkarhana/target-kernel-similar"),
                SimpleNamespace(ref="other/target-kernel"),
            ]

    assert resolver(Api(), "indarkarhana/target-kernel") == []

    class PresentApi(Api):
        def kernels_list(self, **kwargs):
            return [SimpleNamespace(ref="indarkarhana/target-kernel")]

    assert resolver(PresentApi(), "indarkarhana/target-kernel") == [
        "indarkarhana/target-kernel"
    ]


def test_wrapper_uses_sdk_ref_and_verifies_post_push_before_start_and_reconcile():
    source = Path("scripts/run-phase2-cpu-acceptance.ps1").read_text(encoding="utf-8")
    assert "scripts/get-kaggle-kernel-state.py" in source
    assert '"kernels", "pull"' in source
    assert "$kernelRef = [string]$kernelState.next_kernel_ref" in source
    assert '"$kernelSlug/$version"' not in source
    assert "current_kernel_ref" not in source
    assert "Get-CanonicalTextSha256" in source
    assert "[BitConverter]::ToString" in source
    assert "[Convert]::ToHexString" not in source
    push = source.index('"kernels", "push", "-p", $kernelStage')
    post_push = source.index("$postPush = Get-OwnedKernelState")
    verify = source.index("Assert-OwnedKernelServerState $postPush")
    start = source.index("cpu-acceptance start $runId")
    reconcile = source.index("cpu-acceptance reconcile $runId")
    assert push < post_push < verify < start < reconcile
