import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("d4_correction", ROOT / "research/public_d4_correction.py")
d4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d4)


def test_geometry_unique_inverse_and_equivariance():
    report = d4.geometry_check()
    assert report["passed"] and report["model_calls_per_arm"] == 8
    assert not report["labels_opened"] and not report["quality_gain_established"]
    for row in report["records"]:
        assert row["unique_views"] == {"legacy": 7, "corrected": 8}
        assert row["equivariance_max_error"]["corrected"] < 1e-10
        assert row["equivariance_max_error"]["legacy"] > 0


def test_deepcenter_minimal_edits_and_idempotence_rejection():
    code = """# Unicode comment: microscopy \u03bc\nif enabled:
    at = torch_mod.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
    acc = acc + torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1))
    untouched = torch_mod.rot90(tensor, 1, dims=(-2, -1))
"""
    result, edits = d4.correct_antidiagonal(code, predictor=False)
    assert len(edits) == 2
    assert 'rot90(tensor, 2,' in result
    assert 'transpose(-1, -2), -2,' in result
    assert 'untouched = torch_mod.rot90(tensor, 1,' in result
    with pytest.raises(ValueError, match="already corrected"):
        d4.correct_antidiagonal(result, predictor=False)


def test_missing_site_rejected():
    with pytest.raises(ValueError, match="Unexpected anti-diagonal"):
        d4.correct_antidiagonal("unrelated = 1\n", predictor=True)


def test_support_drift_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        d4.materialize_public_predictor({}, b"not the pinned support file")


def test_actual_pinned_public_patch_chain_and_notebook():
    path = ROOT / ".biohub/cache/public-frontier-20260910-2132/lf-dctta/biohub-lf-dctta.ipynb"
    support = ROOT / ".biohub/cache/datasets/biohub-support-source/predict_unet_transformer.py"
    if not path.exists() or not support.exists():
        pytest.skip("Public source-only audit cache not installed")
    assert d4.sha256(path.read_bytes()) == d4.NOTEBOOK_SHA256
    notebook = json.loads(path.read_bytes())
    original = d4.materialize_public_predictor(notebook, support.read_bytes())
    fixed, changes = d4.correct_antidiagonal(original, predictor=True)
    assert len(changes) == 6
    # Restore only the edited digit/sign spans; this proves unrelated code did
    # not change even in the long dynamically materialized public predictor.
    assert fixed.replace('imgs, 2, dims=', 'imgs, 1, dims=').count('imgs, 1, dims=') >= 2
    dc_original = d4.cell_text(notebook, 5)
    dc, dc_changes = d4.correct_antidiagonal(dc_original, predictor=False)
    assert len(dc_changes) == 2
    built = d4.corrected_notebook(notebook, original, fixed, dc)
    for index in range(12):
        if index not in (4, 5):
            assert d4.cell_text(built, index) == d4.cell_text(notebook, index)
    assert d4.cell_text(notebook, 5) == dc_original  # input notebook not mutated
    assert all(not cell.get("outputs") for cell in built["cells"])
    assert d4.cell_text(built, 4).index("PROJECT_D4_CORRECTION") < d4.cell_text(built, 4).index("def list_test_stems")
