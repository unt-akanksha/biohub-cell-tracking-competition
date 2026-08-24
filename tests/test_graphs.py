from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from biohub_tracker.evidence import EvidenceError, PredictionSetClaim, resolve_producer
from biohub_tracker.graphs import (
    GraphData,
    GraphNode,
    GraphValidationError,
    artifact_tree_sha256,
    preflight_prediction_set,
    validate_graph_data,
)
from biohub_tracker.io import canonical_json_bytes, sha256_bytes
from biohub_tracker.ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    completed_payload,
    registration_payload,
    start_payload,
)
from biohub_tracker.manifests import EvaluationManifest, FoldRecord, SampleRecord, write_manifest


def _sample(sample_id="6bba_fov-a"):
    embryo, fov = sample_id.split("_", 1)
    token = sha256_bytes(sample_id.encode())
    return SampleRecord(
        sample_id=sample_id,
        embryo_id=embryo,
        field_of_view_id=fov,
        image_relpath=f"{sample_id}.zarr",
        truth_relpath=f"{sample_id}.geff",
        shape_tzyx=(4, 8, 16, 16),
        dtype="uint16",
        chunks_tzyx=(1, 4, 8, 8),
        scale_zyx_um=("1.625", "0.40625", "0.40625"),
        estimated_number_of_nodes="10",
        gt_node_count=3,
        gt_edge_count=2,
        image_metadata_sha256=token,
        geff_tree_sha256=sha256_bytes((sample_id + "-truth").encode()),
    )


def test_prediction_set_fixture_has_canonical_inventory_hash():
    value = json.loads(Path("tests/fixtures/manifest/prediction-set.json").read_text())
    claim = PredictionSetClaim.from_dict(value)
    assert claim.graph_inventory_sha256 == claim.computed_inventory_sha256()


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


def _manifest(path: Path):
    records = tuple(_sample(name) for name in ("44b6_fov-a", "44b6_fov-b", "6bba_fov-a", "6bba_fov-b"))
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
        scorer_lock_sha256="1" * 64,
        source_inventory_sha256="2" * 64,
        samples=records,
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
    write_manifest(path, manifest)
    return manifest


def _metrics():
    return {
        "pooled": {
            "adjusted_edge_jaccard": 1,
            "edge_jaccard": 1,
            "division_jaccard": 1,
            "node_recall": 1,
        },
        "division_counts": {"tp": 1, "fp": 0, "fn": 0},
        "by_embryo": {"44b6": {}, "6bba": {}},
        "by_fold": {"fold": {}},
        "worst_movie_delta": 0,
    }


def _claim_and_ledger(tmp_path, *, terminal=True, eligible=True, mutate_registration=False, mutate_terminal=False):
    manifest = _manifest(tmp_path / "manifest.json")
    fold = manifest.folds[0]
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    graphs = []
    for sample_id in fold.evaluation_membership:
        graph = pred_dir / f"{sample_id}.geff"
        graph.mkdir()
        (graph / "payload.bin").write_bytes(sample_id.encode())
        graphs.append(
            {
                "sample_id": sample_id,
                "path": graph.name,
                "sha256": artifact_tree_sha256(graph),
                "producer_run_id": "producer-a",
                "fold_id": fold.fold_id,
            }
        )
    inventory_sha = sha256_bytes(canonical_json_bytes(graphs))
    base = registration_payload(
        hypothesis="producer",
        parent=None,
        config={"lr": "0.1"},
        seeds=[1],
        split=fold.fold_id,
        declared_max_runtime_hours="1",
        code={"git_head": "abc"},
    )
    lineage = {
        "manifest_sha256": manifest.manifest_sha256,
        "fold_id": fold.fold_id,
        "train_membership_sha256": fold.train_membership_sha256,
        "calibration_membership_sha256": fold.calibration_membership_sha256,
        "evaluation_membership_sha256": fold.evaluation_membership_sha256,
        "model_sha256": "3" * 64,
        "config_sha256": base["config_sha256"],
        "code_sha256": "4" * 64,
        "data_sha256": "5" * 64,
    }
    claim_value = {
        "schema_version": 1,
        "producer_run_id": "producer-a",
        **lineage,
        "graphs": graphs,
        "graph_inventory_sha256": inventory_sha,
        "artifact_hashes": {"model": "6" * 64, "predictions": "7" * 64},
    }
    claim_path = tmp_path / "prediction-set.json"
    claim_path.write_text(json.dumps(claim_value, sort_keys=True), encoding="utf-8")
    ledger = Ledger(tmp_path / "experiments" / "events.jsonl", tmp_path)
    registered_lineage = dict(lineage)
    if mutate_registration:
        registered_lineage["model_sha256"] = "8" * 64
    registered = registration_payload(
        hypothesis="producer",
        parent=None,
        config={"lr": "0.1"},
        seeds=[1],
        split=fold.fold_id,
        declared_max_runtime_hours="1",
        code={"git_head": "abc"},
        producer_evidence=registered_lineage,
    )
    ledger.append(ExperimentEvent.create("producer-a", EventType.REGISTERED, registered))
    if terminal:
        ledger.append(
            ExperimentEvent.create(
                "producer-a",
                EventType.STARTED,
                start_payload(kaggle_ref="owner/kernel", authorization_id="auth", quota_before_hours="30"),
            )
        )
        terminal_evidence = {
            "evidence_eligible": True,
            "graph_inventory_sha256": "9" * 64 if mutate_terminal else inventory_sha,
            "artifact_hashes": claim_value["artifact_hashes"],
        }
        payload = completed_payload(
            actual_runtime_hours="1",
            quota_after_hours="29",
            metrics=_metrics(),
            producer_evidence=terminal_evidence if eligible else None,
        )
        ledger.append(ExperimentEvent.create("producer-a", EventType.COMPLETED, payload))
    return manifest, pred_dir, claim_path, ledger


def test_native_subvoxel_is_valid_but_submission_requires_integral_coordinates():
    sample = _sample()
    graph = GraphData(
        nodes=(
            GraphNode(10, 0, 1.25, 2.5, 3.75),
            GraphNode(11, 1, 1.5, 2.75, 4.25),
        ),
        edges=((10, 11),),
    )
    assert validate_graph_data(graph, sample, mode="native").node_count == 2
    with pytest.raises(GraphValidationError) as error:
        validate_graph_data(graph, sample, mode="submission")
    assert error.value.reason_code == "NONINTEGRAL_COORDINATE"


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (GraphData((GraphNode(1.5, 0, 1, 1, 1),), ()), "NONINTEGRAL_NODE_ID"),
        (GraphData((GraphNode(1, 0.5, 1, 1, 1),), ()), "NONINTEGRAL_TIME"),
        (GraphData((GraphNode(1, 0, -1, 1, 1),), ()), "SENTINEL_VALUE"),
        (GraphData((GraphNode(1, 0, 1, 1, 1), GraphNode(2, 1, 1, 1, 1)), ((1, 2), (1, 2))), "DUPLICATE_EDGE"),
        (GraphData((GraphNode(1, 0, 1, 1, 1),), ((1, 2),)), "DANGLING_ENDPOINT"),
        (GraphData((GraphNode(1, 0, 1, 1, 1),), ((1, 1),)), "SELF_LOOP"),
        (GraphData((GraphNode(1, 0, 1, 1, 1), GraphNode(2, 2, 1, 1, 1)), ((1, 2),)), "FRAME_STEP_INVALID"),
        (
            GraphData(
                (GraphNode(1, 0, 1, 1, 1), GraphNode(2, 1, 1, 1, 1), GraphNode(3, 0, 1, 1, 1)),
                ((1, 2), (3, 2)),
            ),
            "MERGED_DAUGHTER",
        ),
        (
            GraphData(
                (
                    GraphNode(1, 0, 1, 1, 1),
                    GraphNode(2, 1, 1, 1, 1),
                    GraphNode(3, 1, 1, 1, 1),
                    GraphNode(4, 1, 1, 1, 1),
                ),
                ((1, 2), (1, 3), (1, 4)),
            ),
            "FAKE_FORK",
        ),
        (
            GraphData((GraphNode(1, 0, 1, 1, 1), GraphNode(2, 1, 1, 1, 1)), ((1, 2), (2, 1))),
            "CYCLE",
        ),
    ],
)
def test_graph_integrity_rejections_are_reason_coded(data, reason):
    with pytest.raises(GraphValidationError) as error:
        validate_graph_data(data, _sample(), mode="native")
    assert error.value.reason_code == reason


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("unknown", "UNKNOWN_PRODUCER"),
        ("nonterminal", "NONTERMINAL_PRODUCER"),
        ("ineligible", "PRODUCER_NOT_EVIDENCE_ELIGIBLE"),
        ("registered", "REGISTERED_LINEAGE_MISMATCH"),
        ("terminal", "TERMINAL_ARTIFACT_MISMATCH"),
    ],
)
def test_sidecar_cannot_authorize_unknown_nonterminal_or_mismatched_producer(tmp_path, case, reason):
    manifest, pred_dir, claim_path, ledger = _claim_and_ledger(
        tmp_path,
        terminal=case != "nonterminal",
        eligible=case != "ineligible",
        mutate_registration=case == "registered",
        mutate_terminal=case == "terminal",
    )
    if case == "unknown":
        ledger = Ledger(tmp_path / "other" / "events.jsonl", tmp_path)
    with pytest.raises(EvidenceError) as error:
        preflight_prediction_set(
            pred_dir, claim_path, tmp_path / "manifest.json", manifest.folds[0].fold_id, ledger
        )
    assert error.value.reason_code == reason


def test_wrong_fold_mixed_producer_and_graph_reuse_fail_before_loading(tmp_path):
    manifest, pred_dir, claim_path, ledger = _claim_and_ledger(tmp_path)
    called = False
    with pytest.raises(GraphValidationError) as error:
        preflight_prediction_set(
            pred_dir, claim_path, tmp_path / "manifest.json", manifest.folds[1].fold_id, ledger
        )
    assert error.value.reason_code == "FOLD_MISMATCH"
    assert called is False

    value = json.loads(claim_path.read_text())
    value["graphs"][0]["producer_run_id"] = "other"
    value["graph_inventory_sha256"] = sha256_bytes(canonical_json_bytes(value["graphs"]))
    claim_path.write_text(json.dumps(value), encoding="utf-8")
    claim = PredictionSetClaim.from_dict(value)
    # The terminal now intentionally disagrees with the changed inventory; immutable reuse is rejected first.
    with pytest.raises(EvidenceError) as error:
        resolve_producer(claim, ledger)
    assert error.value.reason_code == "TERMINAL_ARTIFACT_MISMATCH"


def test_missing_sidecar_missing_extra_and_stale_graphs_fail_closed(tmp_path):
    manifest, pred_dir, claim_path, ledger = _claim_and_ledger(tmp_path)
    fold_id = manifest.folds[0].fold_id
    with pytest.raises(EvidenceError) as error:
        preflight_prediction_set(
            pred_dir, tmp_path / "absent.json", tmp_path / "manifest.json", fold_id, ledger
        )
    assert error.value.reason_code == "PREDICTION_SET_MISSING"

    first = next(pred_dir.glob("*.geff"))
    hidden = first.with_suffix(".missing")
    first.rename(hidden)
    with pytest.raises(GraphValidationError) as error:
        preflight_prediction_set(pred_dir, claim_path, tmp_path / "manifest.json", fold_id, ledger)
    assert error.value.reason_code == "GRAPH_COVERAGE_MISMATCH"
    hidden.rename(first)

    extra = pred_dir / "6bba_extra.geff"
    extra.mkdir()
    (extra / "payload.bin").write_bytes(b"extra")
    with pytest.raises(GraphValidationError) as error:
        preflight_prediction_set(pred_dir, claim_path, tmp_path / "manifest.json", fold_id, ledger)
    assert error.value.reason_code == "GRAPH_COVERAGE_MISMATCH"
    extra.rename(pred_dir / "extra.ignored")

    before = (first / "payload.bin").read_bytes()
    (first / "payload.bin").write_bytes(before + b"stale")
    with pytest.raises(GraphValidationError) as error:
        preflight_prediction_set(pred_dir, claim_path, tmp_path / "manifest.json", fold_id, ledger)
    assert error.value.reason_code == "STALE_GRAPH_HASH"


def test_mixed_producer_fails_after_authoritative_terminal_inventory_match(tmp_path):
    manifest, pred_dir, claim_path, original = _claim_and_ledger(tmp_path)
    value = json.loads(claim_path.read_text())
    value["graphs"][0]["producer_run_id"] = "other-producer"
    value["graph_inventory_sha256"] = sha256_bytes(canonical_json_bytes(value["graphs"]))
    claim_path.write_text(json.dumps(value), encoding="utf-8")

    ledger = Ledger(tmp_path / "mixed-ledger" / "events.jsonl", tmp_path)
    original_events = original.read_events()
    ledger.append(original_events[0])
    ledger.append(original_events[1])
    ledger.append(
        ExperimentEvent.create(
            "producer-a",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1",
                quota_after_hours="29",
                metrics=_metrics(),
                producer_evidence={
                    "evidence_eligible": True,
                    "graph_inventory_sha256": value["graph_inventory_sha256"],
                    "artifact_hashes": value["artifact_hashes"],
                },
            ),
        )
    )
    with pytest.raises(GraphValidationError) as error:
        preflight_prediction_set(
            pred_dir,
            claim_path,
            tmp_path / "manifest.json",
            manifest.folds[0].fold_id,
            ledger,
        )
    assert error.value.reason_code == "MIXED_PRODUCER"
