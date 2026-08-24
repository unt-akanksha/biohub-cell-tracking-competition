from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from biohub_tracker.cli import main
from biohub_tracker.graphs import artifact_tree_sha256, preflight_prediction_set
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
from biohub_tracker.scorer_lock import verify_scorer_lock
from biohub_tracker.submission_io import (
    RoundTripError,
    SUBMISSION_HEADER,
    roundtrip_prediction_inventory,
    validate_submission_csv,
)


FIXTURE = Path("tests/fixtures/metric/csv_roundtrip/submission.csv")


@pytest.fixture(scope="module")
def verified():
    return verify_scorer_lock(
        Path("config/official-scorer.lock.json"),
        Path(".biohub/vendor/kaggle-cell-tracking-competition"),
        tracksdata_checkout=Path(".biohub/vendor/tracksdata"),
    )


def _graph(verified, path: Path, coordinates, *, offset_ids=False):
    polars = __import__("polars")
    graph = verified.tracksdata.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, polars.Float64, 0.0)
    discarded = (
        graph.add_node({"t": 0, "z": 0.0, "y": 0.0, "x": 0.0}) if offset_ids else None
    )
    ids = []
    for t, z, y, x in coordinates:
        ids.append(graph.add_node({"t": t, "z": z, "y": y, "x": x}))
    for source, target in zip(ids, ids[1:]):
        graph.add_edge(source, target, {})
    if discarded is not None:
        graph.remove_node(discarded)
    graph.to_geff(path)
    return graph


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


def _metrics():
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


def _roundtrip_fixture(tmp_path: Path, verified):
    truth_root = tmp_path / "truth"
    pred_dir = tmp_path / "predictions"
    truth_root.mkdir()
    pred_dir.mkdir()
    sample_names = ("44b6_fov-a", "6bba_fov-a")
    truth_coordinates = ((0, 1.0, 2.0, 3.0), (1, 2.0, 3.0, 4.0), (2, 3.0, 4.0, 5.0))
    for index, name in enumerate(sample_names):
        coordinates = (
            truth_coordinates
            if index == 1
            else tuple((t, z, y, x + 1.0) for t, z, y, x in truth_coordinates)
        )
        _graph(verified, truth_root / f"{name}.geff", coordinates)
    native_coordinates = ((0, 1.49, 2.49, 3.49), (1, 2.5, 3.5, 4.5), (2, 3.49, 4.49, 5.49))
    _graph(verified, pred_dir / "6bba_fov-a.geff", native_coordinates, offset_ids=True)

    records = []
    for name in sample_names:
        embryo, fov = name.split("_", 1)
        records.append(
            SampleRecord(
                sample_id=name,
                embryo_id=embryo,
                field_of_view_id=fov,
                image_relpath=f"{name}.zarr",
                truth_relpath=f"{name}.geff",
                shape_tzyx=(4, 8, 16, 16),
                dtype="uint16",
                chunks_tzyx=(1, 4, 8, 8),
                scale_zyx_um=("1.625", "0.40625", "0.40625"),
                estimated_number_of_nodes="3",
                gt_node_count=3,
                gt_edge_count=2,
                image_metadata_sha256=sha256_bytes((name + "-image").encode()),
                geff_tree_sha256=artifact_tree_sha256(truth_root / f"{name}.geff"),
            )
        )
    sample_map = {item.sample_id: item for item in records}
    folds = []
    for train, evaluate in (("44b6", "6bba"), ("6bba", "44b6")):
        train_ids = (f"{train}_fov-a",)
        eval_ids = (f"{evaluate}_fov-a",)
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
                train_membership_sha256=_membership_sha(sample_map, train_ids),
                calibration_membership_sha256=_membership_sha(sample_map, train_ids),
                evaluation_membership_sha256=_membership_sha(sample_map, eval_ids),
            )
        )
    manifest = EvaluationManifest(
        scorer_lock_sha256=verified.lock_sha256,
        source_inventory_sha256="1" * 64,
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
    fold = manifest.folds[0]
    graph_record = {
        "sample_id": "6bba_fov-a",
        "path": "6bba_fov-a.geff",
        "sha256": artifact_tree_sha256(pred_dir / "6bba_fov-a.geff"),
        "producer_run_id": "producer-roundtrip",
        "fold_id": fold.fold_id,
    }
    inventory_sha = sha256_bytes(canonical_json_bytes([graph_record]))
    base = registration_payload(
        hypothesis="roundtrip fixture",
        parent=None,
        config={"candidate": "fixture"},
        seeds=[1],
        split=fold.fold_id,
        declared_max_runtime_hours="1",
        code={"git_head": "fixture"},
    )
    lineage = {
        "manifest_sha256": manifest.manifest_sha256,
        "fold_id": fold.fold_id,
        "train_membership_sha256": fold.train_membership_sha256,
        "calibration_membership_sha256": fold.calibration_membership_sha256,
        "evaluation_membership_sha256": fold.evaluation_membership_sha256,
        "model_sha256": "2" * 64,
        "config_sha256": base["config_sha256"],
        "code_sha256": "3" * 64,
        "data_sha256": "4" * 64,
    }
    artifact_hashes = {"model": "5" * 64, "native_predictions": "6" * 64}
    claim = {
        "schema_version": 1,
        "producer_run_id": "producer-roundtrip",
        **lineage,
        "graphs": [graph_record],
        "graph_inventory_sha256": inventory_sha,
        "artifact_hashes": artifact_hashes,
    }
    claim_path = tmp_path / "prediction-set.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    ledger = Ledger(tmp_path / "experiments" / "events.jsonl", tmp_path)
    ledger.append(
        ExperimentEvent.create(
            "producer-roundtrip",
            EventType.REGISTERED,
            registration_payload(
                hypothesis="roundtrip fixture",
                parent=None,
                config={"candidate": "fixture"},
                seeds=[1],
                split=fold.fold_id,
                declared_max_runtime_hours="1",
                code={"git_head": "fixture"},
                producer_evidence=lineage,
            ),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "producer-roundtrip",
            EventType.STARTED,
            start_payload(kaggle_ref="owner/kernel", authorization_id="auth", quota_before_hours="30"),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "producer-roundtrip",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1",
                quota_after_hours="29",
                metrics=_metrics(),
                producer_evidence={
                    "evidence_eligible": True,
                    "graph_inventory_sha256": inventory_sha,
                    "artifact_hashes": artifact_hashes,
                },
            ),
        )
    )
    inventory = preflight_prediction_set(pred_dir, claim_path, manifest_path, fold.fold_id, ledger)
    return inventory, truth_root, pred_dir / "6bba_fov-a.geff"


def test_checked_in_csv_fixture_is_canonical():
    result = validate_submission_csv(FIXTURE, ["6bba_fixture"])
    assert result.row_count == 3
    assert result.node_counts == (("6bba_fixture", 2),)
    assert result.edge_counts == (("6bba_fixture", 1),)
    with pytest.raises(RoundTripError) as error:
        validate_submission_csv(FIXTURE, ["44b6_missing", "6bba_fixture"])
    assert error.value.reason_code == "CSV_COVERAGE_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("header", "CSV_HEADER_MISMATCH"),
        ("placeholder", "CSV_PLACEHOLDER_INVALID"),
        ("duplicate", "CSV_DUPLICATE_NODE_ID"),
        ("dangling", "CSV_DANGLING_EDGE"),
        ("noninteger", "CSV_NONINTEGER_FIELD"),
        ("row_order", "CSV_ROW_ORDER_INVALID"),
    ],
)
def test_csv_schema_id_placeholder_and_topology_drift_are_rejected(tmp_path, mutation, reason):
    rows = list(csv.DictReader(FIXTURE.open(newline="", encoding="utf-8")))
    header = list(SUBMISSION_HEADER)
    if mutation == "header":
        header[3], header[4] = header[4], header[3]
    elif mutation == "placeholder":
        rows[2]["z"] = "0"
    elif mutation == "duplicate":
        rows[1]["node_id"] = "10"
    elif mutation == "dangling":
        rows[2]["target_id"] = "99"
    elif mutation == "noninteger":
        rows[0]["z"] = "1.5"
    elif mutation == "row_order":
        rows[1], rows[2] = rows[2], rows[1]
        for index, row in enumerate(rows):
            row["id"] = str(index)
    path = tmp_path / "submission.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RoundTripError) as error:
        validate_submission_csv(path, ["6bba_fixture"])
    assert error.value.reason_code == reason


def test_integer_roundtrip_is_authoritative_deterministic_and_lineage_bound(tmp_path, verified):
    inventory, truth_root, native_path = _roundtrip_fixture(tmp_path, verified)
    native_before = artifact_tree_sha256(native_path)
    first = roundtrip_prediction_inventory(inventory, verified, truth_root, tmp_path / "first")
    second = roundtrip_prediction_inventory(inventory, verified, truth_root, tmp_path / "second")

    assert first.semantic_parity is True
    assert first.official_counts_parity is True
    assert first.source_space == "native-nonauthoritative"
    assert first.authoritative_space == "integer-csv-rebuilt-geff"
    assert first.source_lineage["producer_run_id"] == inventory.claim.producer_run_id
    assert first.source_lineage["artifact_hashes"] == dict(inventory.claim.artifact_hashes)
    assert first.source_lineage["manifest_sha256"] == inventory.manifest.manifest_sha256
    assert artifact_tree_sha256(native_path) == native_before
    assert (tmp_path / "first" / "submission.csv").read_bytes() == (
        tmp_path / "second" / "submission.csv"
    ).read_bytes()
    assert first.evidence_sha256 == second.evidence_sha256

    rows = list(csv.DictReader((tmp_path / "first" / "submission.csv").open()))
    node_rows = [row for row in rows if row["row_type"] == "node"]
    assert [(row["z"], row["y"], row["x"]) for row in node_rows] == [
        ("1", "2", "3"),
        ("2", "4", "4"),
        ("3", "4", "5"),
    ]
    assert [row["node_id"] for row in node_rows] == ["1", "2", "3"]
    assert (tmp_path / "first" / "submission-space-inventory.json").is_file()
    assert (tmp_path / "first" / "graphs" / "6bba_fov-a.geff").is_dir()


def test_graph_and_roundtrip_cli_publish_provenance_only_after_all_gates(tmp_path, verified, capsys):
    inventory, truth_root, _ = _roundtrip_fixture(tmp_path, verified)
    scorer_lock = Path("config/official-scorer.lock.json").resolve()
    scorer_checkout = Path(".biohub/vendor/kaggle-cell-tracking-competition").resolve()
    tracksdata_checkout = Path(".biohub/vendor/tracksdata").resolve()
    common = [
        "--root",
        str(tmp_path),
        "--pred-dir",
        str(inventory.pred_dir),
        "--producer-manifest",
        str(tmp_path / "prediction-set.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--fold",
        inventory.fold.fold_id,
        "--ledger",
        str(tmp_path / "experiments" / "events.jsonl"),
        "--scorer-lock",
        str(scorer_lock),
        "--scorer-checkout",
        str(scorer_checkout),
        "--tracksdata-checkout",
        str(tracksdata_checkout),
    ]
    graph_args = common[:2] + ["graph", "validate"] + common[2:] + ["--mode", "native"]
    assert main(graph_args) == 0
    graph_output = json.loads(capsys.readouterr().out)
    assert graph_output["producer_run_id"] == inventory.claim.producer_run_id
    assert graph_output["graph_inventory_sha256"] == inventory.claim.graph_inventory_sha256

    roundtrip_args = common[:2] + ["submission", "roundtrip"] + common[2:] + [
        "--truth-root",
        str(truth_root),
        "--output-dir",
        str(tmp_path / "cli-roundtrip"),
    ]
    assert main(roundtrip_args) == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["authoritative_space"] == "integer-csv-rebuilt-geff"
    assert evidence["official_counts_parity"] is True
