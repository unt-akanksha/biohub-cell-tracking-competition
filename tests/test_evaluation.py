from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from biohub_tracker.evaluation import (
    ExactEvaluationError,
    ExactEvaluationRequest,
    ExactReport,
    PredictionSetRef,
    evaluate_exact,
    metric_exploit_audit,
    validate_exact_report,
)
from biohub_tracker.graphs import GraphData, GraphNode, artifact_tree_sha256
from biohub_tracker.io import canonical_json_bytes, sha256_bytes
from biohub_tracker.ledger import (
    EventType,
    ExactEvaluationStatus,
    ExperimentEvent,
    Ledger,
    completed_payload,
    exact_evaluation_registration_payload,
    exact_evaluation_started_payload,
    registration_payload,
    reconstruct_exact_evaluations,
    resolved_exact_member,
    start_payload,
)
from biohub_tracker.manifests import EvaluationManifest, FoldRecord, SampleRecord, write_manifest
from biohub_tracker.scorer_lock import ScorerLock, verify_scorer_lock


LOCK = Path("config/official-scorer.lock.json").resolve()
POLICY = Path("config/evaluation-policy.json").resolve()
CHECKOUT = Path(".biohub/vendor/kaggle-cell-tracking-competition").resolve()
TRACKSDATA = Path(".biohub/vendor/tracksdata").resolve()


@pytest.fixture(scope="module")
def verified():
    return verify_scorer_lock(LOCK, CHECKOUT, tracksdata_checkout=TRACKSDATA)


def _graph(verified, path: Path, coordinates):
    polars = __import__("polars")
    graph = verified.tracksdata.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, polars.Float64, 0.0)
    ids = [
        graph.add_node({"t": t, "z": z, "y": y, "x": x})
        for t, z, y, x in coordinates
    ]
    for source, target in zip(ids, ids[1:]):
        graph.add_edge(source, target, {})
    graph.to_geff(path)


def _membership_sha(samples, members):
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "sample_id": name,
                    "image_metadata_sha256": samples[name].image_metadata_sha256,
                    "geff_tree_sha256": samples[name].geff_tree_sha256,
                }
                for name in sorted(members)
            ]
        )
    )


def _legacy_metrics():
    return {
        "pooled": {
            "adjusted_edge_jaccard": 1,
            "edge_jaccard": 1,
            "division_jaccard": 0,
            "node_recall": 1,
        },
        "division_counts": {"tp": 0, "fp": 0, "fn": 0},
        "by_embryo": {"44b6": {}, "6bba": {}},
        "by_fold": {"fold": {}},
        "worst_movie_delta": 0,
    }


def _fixture(tmp_path: Path, verified):
    truth = tmp_path / "truth"
    truth.mkdir()
    specs = {
        "44b6_long": tuple((t, 1.0, 2.0, float(t + 2)) for t in range(6)),
        "44b6_short": ((0, 2.0, 3.0, 4.0), (1, 2.0, 3.0, 5.0)),
        "6bba_long": tuple((t, 3.0, 4.0, float(t + 3)) for t in range(5)),
        "6bba_short": ((0, 4.0, 5.0, 6.0), (1, 4.0, 5.0, 7.0)),
    }
    for name, coordinates in specs.items():
        _graph(verified, truth / f"{name}.geff", coordinates)
    records = []
    for name, coordinates in sorted(specs.items()):
        embryo, fov = name.split("_", 1)
        records.append(
            SampleRecord(
                sample_id=name,
                embryo_id=embryo,
                field_of_view_id=fov,
                image_relpath=f"{name}.zarr",
                truth_relpath=f"{name}.geff",
                shape_tzyx=(8, 16, 32, 32),
                dtype="uint16",
                chunks_tzyx=(1, 8, 16, 16),
                scale_zyx_um=("1.625", "0.40625", "0.40625"),
                estimated_number_of_nodes=str(len(coordinates)),
                gt_node_count=len(coordinates),
                gt_edge_count=len(coordinates) - 1,
                image_metadata_sha256=sha256_bytes((name + "-image").encode()),
                geff_tree_sha256=artifact_tree_sha256(truth / f"{name}.geff"),
            )
        )
    samples = {item.sample_id: item for item in records}
    folds = []
    for train, evaluate in (("44b6", "6bba"), ("6bba", "44b6")):
        train_ids = tuple(name for name in samples if name.startswith(train))
        eval_ids = tuple(name for name in samples if name.startswith(evaluate))
        folds.append(
            FoldRecord(
                fold_id=f"fold-{train}-to-{evaluate}",
                train_embryo_id=train,
                evaluation_embryo_id=evaluate,
                train_membership=train_ids,
                calibration_membership=train_ids,
                threshold_selection_membership=train_ids,
                early_stopping_membership=train_ids,
                evaluation_membership=eval_ids,
                train_membership_sha256=_membership_sha(samples, train_ids),
                calibration_membership_sha256=_membership_sha(samples, train_ids),
                evaluation_membership_sha256=_membership_sha(samples, eval_ids),
            )
        )
    manifest = EvaluationManifest(
        scorer_lock_sha256=verified.lock_sha256,
        source_inventory_sha256="f" * 64,
        samples=tuple(records),
        folds=tuple(folds),
        overlap_audit={
            "cross_side_artifact_hashes": [],
            "cross_side_paths": [],
            "duplicate_sample_ids": [],
            "evaluation_union_complete": True,
            "movie_splitting": False,
            "passed": True,
        },
        manifest_sha256="",
        created_at="2026-08-24T00:00:00Z",
    )
    manifest = replace(manifest, manifest_sha256=manifest.computed_sha256())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    ledger = Ledger(tmp_path / "experiments/events.jsonl", tmp_path)
    refs = {"baseline": [], "candidate": []}
    producer_slots = []
    for role in ("baseline", "candidate"):
        for slot, fold in enumerate(manifest.folds):
            run_id = f"{role}-{slot}"
            pred_dir = tmp_path / "predictions" / role / fold.fold_id
            pred_dir.mkdir(parents=True)
            graphs = []
            for sample_id in fold.evaluation_membership:
                coordinates = specs[sample_id]
                if role == "candidate" and sample_id == "6bba_short":
                    coordinates = (coordinates[0], (1, 4.0, 5.0, 25.0))
                path = pred_dir / f"{sample_id}.geff"
                _graph(verified, path, coordinates)
                graphs.append(
                    {
                        "sample_id": sample_id,
                        "path": path.name,
                        "sha256": artifact_tree_sha256(path),
                        "producer_run_id": run_id,
                        "fold_id": fold.fold_id,
                    }
                )
            inventory_sha = sha256_bytes(canonical_json_bytes(graphs))
            config = {"role": role, "fold": fold.fold_id}
            base = registration_payload(
                hypothesis=run_id,
                parent=None,
                config=config,
                seeds=[slot],
                split=fold.fold_id,
                declared_max_runtime_hours="1",
                code={"git_head": run_id},
            )
            token = sha256_bytes(run_id.encode())
            lineage = {
                "manifest_sha256": manifest.manifest_sha256,
                "fold_id": fold.fold_id,
                "train_membership_sha256": fold.train_membership_sha256,
                "calibration_membership_sha256": fold.calibration_membership_sha256,
                "evaluation_membership_sha256": fold.evaluation_membership_sha256,
                "model_sha256": token,
                "config_sha256": base["config_sha256"],
                "code_sha256": token,
                "data_sha256": token,
            }
            artifacts = {"predictions": inventory_sha, "model": token}
            claim = {
                "schema_version": 1,
                "producer_run_id": run_id,
                **lineage,
                "graphs": graphs,
                "graph_inventory_sha256": inventory_sha,
                "artifact_hashes": artifacts,
            }
            sidecar = pred_dir / "prediction-set.json"
            sidecar.write_text(json.dumps(claim, sort_keys=True), encoding="utf-8")
            ledger.append(
                ExperimentEvent.create(
                    run_id,
                    EventType.REGISTERED,
                    registration_payload(
                        hypothesis=run_id,
                        parent=None,
                        config=config,
                        seeds=[slot],
                        split=fold.fold_id,
                        declared_max_runtime_hours="1",
                        code={"git_head": run_id},
                        producer_evidence=lineage,
                    ),
                )
            )
            ledger.append(
                ExperimentEvent.create(
                    run_id,
                    EventType.STARTED,
                    start_payload(kaggle_ref="owner/kernel", authorization_id=run_id, quota_before_hours="30"),
                )
            )
            ledger.append(
                ExperimentEvent.create(
                    run_id,
                    EventType.COMPLETED,
                    completed_payload(
                        actual_runtime_hours="1",
                        quota_after_hours="29",
                        metrics=_legacy_metrics(),
                        producer_evidence={
                            "evidence_eligible": True,
                            "graph_inventory_sha256": inventory_sha,
                            "artifact_hashes": artifacts,
                        },
                    ),
                )
            )
            refs[role].append(PredictionSetRef(fold.fold_id, pred_dir, sidecar))
            producer_slots.append((role, fold.fold_id, run_id))
    members = [
        resolved_exact_member(ledger.read_events(), role=role, fold_id=fold, producer_run_id=run)
        for role, fold, run in producer_slots
    ]
    members.sort(key=lambda item: (item["role"], item["fold_id"]))
    policy_sha = sha256_bytes(canonical_json_bytes(json.loads(POLICY.read_text())))
    ledger.append(
        ExperimentEvent.create(
            "eval-fixture",
            EventType.EXACT_EVALUATION_REGISTERED,
            exact_evaluation_registration_payload(
                evaluation_run_id="eval-fixture",
                scorer_lock_sha256=verified.lock_sha256,
                environment_lock_sha256=verified.lock.environment_lock_sha256,
                manifest_sha256=manifest.manifest_sha256,
                evaluation_policy_sha256=policy_sha,
                evidence_kind="synthetic_fixture",
                members=members,
            ),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "eval-fixture",
            EventType.EXACT_EVALUATION_STARTED,
            exact_evaluation_started_payload(evaluation_run_id="eval-fixture"),
        )
    )
    return manifest_path, truth, ledger, refs


def _request(tmp_path: Path, verified) -> tuple[ExactEvaluationRequest, Ledger]:
    manifest_path, truth, ledger, refs = _fixture(tmp_path, verified)
    return (
        ExactEvaluationRequest(
            ledger_path=ledger.path,
            evaluation_run_id="eval-fixture",
            truth_dir=truth,
            manifest_path=manifest_path,
            scorer_lock_path=LOCK,
            evaluation_policy_path=POLICY,
            baseline_sets=tuple(refs["baseline"]),
            candidate_sets=tuple(refs["candidate"]),
            output_dir=tmp_path / "exact-output",
            scorer_checkout=CHECKOUT,
            tracksdata_checkout=TRACKSDATA,
            workspace_root=tmp_path,
        ),
        ledger,
    )


def test_exact_canonical_complete_movie_report_is_pooled_and_ledger_attached(tmp_path, verified):
    request, ledger = _request(tmp_path, verified)
    report = evaluate_exact(request)
    assert report.core["evidence_kind"] == "synthetic_fixture"
    assert report.core["coverage"]["complete"] is True
    assert len(report.core["official"]["baseline"]["by_movie"]) == 4
    assert set(report.core["official"]["candidate"]["by_embryo"]) == {"44b6", "6bba"}
    assert report.core["official"]["candidate"]["pooled"]["score"] != report.core["official"]["baseline"]["pooled"]["score"]
    assert report.core["diagnostics"]["authority"] == "non_authoritative_diagnostic"
    assert report.core["diagnostics"]["organizer_input_eligible"] is False
    assert report.core["diagnostics"]["candidate"]["pooled"]["reconciliation"]["status"] == "passed"
    assert "diagnostic_state" not in json.dumps(report.core["official"], sort_keys=True)
    assert report.core["integrity_checks"]["authoritative_prediction_space"] == "integer-csv-rebuilt-geff"
    assert report.core["integrity_checks"]["metric_exploit_audit"] == "passed"
    assert report.core["integrity_checks"]["metric_exploit_evidence"]["audit_sha256"]
    assert validate_exact_report(
        ExactReport.from_files(report.core_path, report.envelope_path),
        ledger_path=ledger.path,
        workspace_root=tmp_path,
    ).core_sha256 == report.core_sha256


def test_production_metric_exploit_audit_detects_frozen_shifted_signature():
    lock = ScorerLock.from_dict(json.loads(LOCK.read_text(encoding="utf-8")))
    fixture_path = LOCK.parent.parent / lock.patched_exploit_path
    case = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]["hack2"]
    graph = GraphData(
        nodes=tuple(
            GraphNode(
                node["id"],
                node["t"],
                node["z"] + 100,
                node["y"] + 100,
                node["x"] + 100,
            )
            for node in case["prediction"]["nodes"]
        ),
        edges=tuple(tuple(edge) for edge in case["prediction"]["edges"]),
    )
    audit = metric_exploit_audit(
        scorer_lock=lock,
        scorer_lock_sha256=lock.semantic_sha256,
        fixture_path=fixture_path,
        candidate_graphs=[("production-shaped", graph)],
    )
    assert audit["status"] == "failed"
    assert audit["reason_codes"] == ["KNOWN_EXPLOIT_GRAPH_SIGNATURE"]
    assert audit["known_signature_matches"] == [
        {"sample_id": "production-shaped", "fixture_case": "hack2"}
    ]


def test_exact_publication_retries_after_kill_immediately_before_rename(
    tmp_path, verified, monkeypatch
):
    request, ledger = _request(tmp_path, verified)
    import biohub_tracker.evaluation as evaluation_module

    real_rename = evaluation_module.os.rename

    def kill_before_rename(source, destination):
        raise SystemExit("kill before rename")

    monkeypatch.setattr(evaluation_module.os, "rename", kill_before_rename)
    with pytest.raises(SystemExit, match="before rename"):
        evaluate_exact(request)
    assert not request.output_dir.exists()
    monkeypatch.setattr(evaluation_module.os, "rename", real_rename)
    report = evaluate_exact(request)
    assert report.core_path == request.output_dir / "exact-report-core.json"
    assert reconstruct_exact_evaluations(ledger.read_events())["eval-fixture"].status is (
        ExactEvaluationStatus.COMPLETED
    )


def test_exact_publication_recovers_kill_after_rename_before_completion(
    tmp_path, verified, monkeypatch
):
    request, ledger = _request(tmp_path, verified)
    real_append = Ledger.append

    def kill_before_completion(self, event):
        if event.event_type is EventType.EXACT_EVALUATION_COMPLETED:
            raise SystemExit("kill before completion")
        return real_append(self, event)

    monkeypatch.setattr(Ledger, "append", kill_before_completion)
    with pytest.raises(SystemExit, match="before completion"):
        evaluate_exact(request)
    assert request.output_dir.is_dir()
    assert reconstruct_exact_evaluations(ledger.read_events())["eval-fixture"].status is (
        ExactEvaluationStatus.RUNNING
    )
    monkeypatch.setattr(Ledger, "append", real_append)
    report = evaluate_exact(request)
    assert validate_exact_report(
        report, ledger_path=ledger.path, workspace_root=tmp_path
    ).core_sha256 == report.core_sha256


def test_kill_after_rename_cannot_bless_a_rehashed_replacement(
    tmp_path, verified, monkeypatch
):
    request, ledger = _request(tmp_path, verified)
    real_append = Ledger.append

    def kill_before_completion(self, event):
        if event.event_type is EventType.EXACT_EVALUATION_COMPLETED:
            raise SystemExit("kill before completion")
        return real_append(self, event)

    monkeypatch.setattr(Ledger, "append", kill_before_completion)
    with pytest.raises(SystemExit, match="before completion"):
        evaluate_exact(request)
    monkeypatch.setattr(Ledger, "append", real_append)

    core_path = request.output_dir / "exact-report-core.json"
    envelope_path = request.output_dir / "exact-report-envelope.json"
    core = json.loads(core_path.read_text(encoding="utf-8"))
    core["official"]["candidate"]["pooled"]["score"] = "999"
    replacement_core_sha = sha256_bytes(canonical_json_bytes(core))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["report_core_sha256"] = replacement_core_sha
    core_path.write_bytes(canonical_json_bytes(core) + b"\n")
    envelope_path.write_bytes(canonical_json_bytes(envelope) + b"\n")

    with pytest.raises(ExactEvaluationError, match="EXACT_MATERIALIZATION_MISMATCH"):
        evaluate_exact(request)
    assert reconstruct_exact_evaluations(ledger.read_events())["eval-fixture"].status is (
        ExactEvaluationStatus.RUNNING
    )


def test_exact_publication_retries_kill_after_completion_append(
    tmp_path, verified, monkeypatch
):
    request, ledger = _request(tmp_path, verified)
    import biohub_tracker.evaluation as evaluation_module

    real_validate = evaluation_module.validate_exact_report

    def kill_after_completion(report, **kwargs):
        if kwargs.get("require_completed", True):
            raise SystemExit("kill after completion")
        return real_validate(report, **kwargs)

    monkeypatch.setattr(evaluation_module, "validate_exact_report", kill_after_completion)
    with pytest.raises(SystemExit, match="after completion"):
        evaluate_exact(request)
    assert request.output_dir.is_dir()
    assert reconstruct_exact_evaluations(ledger.read_events())["eval-fixture"].status is (
        ExactEvaluationStatus.COMPLETED
    )
    monkeypatch.setattr(evaluation_module, "validate_exact_report", real_validate)
    recovered = evaluate_exact(request)
    assert recovered.core_sha256 == ExactReport.from_files(
        recovered.core_path, recovered.envelope_path
    ).core_sha256


def test_exact_report_unknown_and_nonfinite_fields_fail_closed():
    with pytest.raises(Exception):
        validate_exact_report(
            ExactReport({"schema_version": "biohub.exact-report.v1", "unknown": float("nan")}, {}, "0" * 64, "0" * 64),
            require_completed=False,
        )
