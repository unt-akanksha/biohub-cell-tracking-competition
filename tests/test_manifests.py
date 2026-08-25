from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from biohub_tracker.cli import main
from biohub_tracker.io import canonical_json_bytes
from biohub_tracker.manifests import ManifestError, build_manifest, load_manifest, verify_manifest, write_manifest


FIXTURE = Path("tests/fixtures/manifest/official-metadata.json")
LOCK = Path("config/official-scorer.lock.json")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def materialize(root: Path, *, samples=None) -> Path:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    selected = samples or fixture["samples"]
    axes = [{"name": name, "type": "time" if name == "t" else "space"} for name in ("t", "z", "y", "x")]
    for sample in selected:
        name = sample["sample_id"]
        image = root / f"{name}.zarr"
        truth = root / f"{name}.geff"
        _write_json(image / "zarr.json", {"attributes": {"multiscales": [{"axes": axes, "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, *fixture['scale_zyx_um']]}]}]}]}})
        _write_json(image / "0" / "zarr.json", {"shape": fixture["shape_tzyx"], "data_type": "uint16", "chunk_grid": {"configuration": {"chunk_shape": fixture["chunks_tzyx"]}}})
        geff = {
            "geff_version": "0.5.0",
            "directed": True,
            "axes": axes,
            "node_props_metadata": {key: {} for key in ("t", "z", "y", "x")},
            "edge_props_metadata": {},
            "extra": {"estimated_number_of_nodes": sample["estimated_number_of_nodes"], "sample_id": sample.get("claimed_sample_id", name)},
        }
        _write_json(truth / "zarr.json", {"attributes": {"geff": geff}})
        for relative, shape in {
            "nodes/ids": [sample["node_count"]],
            "edges/ids": [sample["edge_count"], 2],
            "nodes/props/t": [sample["node_count"]],
            "nodes/props/z": [sample["node_count"]],
            "nodes/props/y": [sample["node_count"]],
            "nodes/props/x": [sample["node_count"]],
        }.items():
            _write_json(truth / relative / "zarr.json", {"shape": shape})
        (truth / "fixture-id.txt").write_text(name, encoding="utf-8")
    return root


def test_reciprocal_manifest_is_canonical_complete_and_timestamp_independent(tmp_path):
    data = materialize(tmp_path / "train")
    first = build_manifest(data, LOCK, created_at=datetime(2026, 8, 24, tzinfo=timezone.utc))
    second = build_manifest(data, LOCK, created_at=datetime(2026, 8, 25, tzinfo=timezone.utc))

    assert [fold.fold_id for fold in first.folds] == ["fold-44b6-to-6bba", "fold-6bba-to-44b6"]
    assert sorted(item for fold in first.folds for item in fold.evaluation_membership) == [sample.sample_id for sample in first.samples]
    assert all(set(fold.calibration_membership) <= set(fold.train_membership) for fold in first.folds)
    assert first.manifest_sha256 == second.manifest_sha256
    assert canonical_json_bytes(first.semantic_dict()) == canonical_json_bytes(second.semantic_dict())
    assert first.to_dict()["evidence"] != second.to_dict()["evidence"]
    assert first.overlap_audit["passed"] is True


def test_live_ome_zarr_uppercase_axis_names_normalize_to_tzyx(tmp_path):
    data = materialize(tmp_path / "train")
    units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    for metadata_path in data.glob("*.zarr/zarr.json"):
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        axes = value["attributes"]["multiscales"][0]["axes"]
        for axis in axes:
            original = axis["name"]
            axis["name"] = original.upper()
            axis["unit"] = units[original]
        _write_json(metadata_path, value)

    manifest = build_manifest(data, LOCK)
    assert manifest.samples
    assert all(sample.shape_tzyx for sample in manifest.samples)


def test_membership_hashes_change_with_corresponding_identity(tmp_path):
    original = build_manifest(materialize(tmp_path / "a"), LOCK)
    changed_root = materialize(tmp_path / "b")
    (changed_root / "44b6_fov-a.geff" / "fixture-id.txt").write_text("changed", encoding="utf-8")
    changed = build_manifest(changed_root, LOCK)
    original_by_fold = {fold.fold_id: fold for fold in original.folds}
    changed_by_fold = {fold.fold_id: fold for fold in changed.folds}
    forward = original_by_fold["fold-44b6-to-6bba"]
    changed_forward = changed_by_fold[forward.fold_id]
    reverse = original_by_fold["fold-6bba-to-44b6"]
    changed_reverse = changed_by_fold[reverse.fold_id]
    assert forward.train_membership_sha256 != changed_forward.train_membership_sha256
    assert forward.calibration_membership_sha256 != changed_forward.calibration_membership_sha256
    assert reverse.evaluation_membership_sha256 != changed_reverse.evaluation_membership_sha256


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_manifest_rejects_geff_child_symlink_escapes(tmp_path, target_kind):
    data = materialize(tmp_path / "train")
    outside = tmp_path / "outside"
    outside.mkdir()
    if target_kind == "file":
        target = outside / "secret.bin"
        target.write_bytes(b"not official data")
    else:
        target = outside / "secret-tree"
        target.mkdir()
        (target / "secret.bin").write_bytes(b"not official data")
    link = data / "44b6_fov-a.geff" / f"escape-{target_kind}"
    try:
        os.symlink(target, link, target_is_directory=target_kind == "directory")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ManifestError) as error:
        build_manifest(data, LOCK)
    assert error.value.reason_code == "PATH_ESCAPE"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("unknown", "UNKNOWN_IDENTITY"),
        ("unpaired", "UNPAIRED_ARTIFACT"),
        ("conflicting", "IDENTITY_CONFLICT"),
        ("duplicate_hash", "DUPLICATE_ARTIFACT_HASH"),
    ],
)
def test_identity_overlap_and_pairing_fail_closed_without_output(tmp_path, mutation, reason):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    samples = [dict(item) for item in fixture["samples"]]
    if mutation == "unknown":
        samples[0]["sample_id"] = "other_fov-a"
    if mutation == "conflicting":
        samples[0]["claimed_sample_id"] = "6bba_fov-a"
    if mutation == "duplicate_hash":
        samples[0]["claimed_sample_id"] = None
        samples[2]["claimed_sample_id"] = None
    data = materialize(tmp_path / "train", samples=samples)
    if mutation == "unpaired":
        # A mismatched stem is sufficient to exercise the fail-closed pairing
        # check and avoids flaky directory renames under Windows file scanners.
        (data / "unpaired.geff").mkdir()
    if mutation == "duplicate_hash":
        left = data / "44b6_fov-a.geff"
        right = data / "6bba_fov-a.geff"
        for target in right.rglob("*"):
            if target.is_file():
                relative = target.relative_to(right)
                source = left / relative
                target.write_bytes(source.read_bytes())
    output = tmp_path / "manifest.json"
    with pytest.raises(ManifestError) as error:
        write_manifest(output, build_manifest(data, LOCK))
    assert error.value.reason_code == reason
    assert not output.exists()


def test_calibration_on_evaluation_side_and_overlap_tamper_are_rejected(tmp_path):
    data = materialize(tmp_path / "train")
    with pytest.raises(ManifestError) as error:
        build_manifest(data, LOCK, calibration_by_fold={"fold-44b6-to-6bba": ["6bba_fov-a"]})
    assert error.value.reason_code == "CALIBRATION_LEAKAGE"

    manifest = build_manifest(data, LOCK)
    value = manifest.to_dict()
    value["folds"][0]["evaluation_membership"] = list(value["folds"][0]["train_membership"])
    path = tmp_path / "tampered.json"
    _write_json(path, value)
    with pytest.raises(ManifestError) as error:
        load_manifest(path)
    assert error.value.reason_code in {"SOURCE_OVERLAP", "IDENTITY_CONFLICT"}


def test_manifest_cli_build_verify_is_immutable_and_no_real_fixture_claim(tmp_path, capsys):
    accepted_path = Path("manifests/reciprocal-embryo-v1.json")
    accepted_before = accepted_path.read_bytes()
    data = materialize(tmp_path / "train")
    output = tmp_path / "fixture-manifest.json"
    assert main(["manifest", "build", "--data-root", str(data), "--output", str(output), "--scorer-lock", str(LOCK)]) == 0
    first = output.read_bytes()
    assert main(["manifest", "verify", "--manifest", str(output), "--data-root", str(data), "--scorer-lock", str(LOCK)]) == 0
    assert verify_manifest(output, data_root=data, scorer_lock_path=LOCK).manifest_sha256
    assert output.read_bytes() == first
    assert accepted_path.read_bytes() == accepted_before
    assert "manifest_sha256" in capsys.readouterr().out
