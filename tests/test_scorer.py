from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from biohub_tracker.io import canonical_json_bytes, sha256_bytes
from biohub_tracker.scorer import run_fixture_tracer
from biohub_tracker.scorer_lock import ScorerVerificationError, verify_scorer_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "official-scorer.lock.json"
CHECKOUT = ROOT / ".biohub" / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA = ROOT / ".biohub" / "vendor" / "tracksdata"
FIXTURE = ROOT / "tests" / "fixtures" / "metric" / "graph_specs" / "perfect-linear.json"
EXPECTED = ROOT / "tests" / "fixtures" / "metric" / "expected" / "official-counts.json"


def _write_lock(tmp_path: Path, mutate) -> Path:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    mutate(value)
    target = tmp_path / "config" / "official-scorer.lock.json"
    target.parent.mkdir(parents=True)
    requirements = tmp_path / "requirements" / "evaluation-lock.txt"
    requirements.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "requirements" / "evaluation-lock.txt", requirements)
    expected = tmp_path / "tests" / "fixtures" / "metric" / "expected" / "official-counts.json"
    expected.parent.mkdir(parents=True)
    shutil.copy2(EXPECTED, expected)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def test_lock_to_official_summary_tracer() -> None:
    verified = verify_scorer_lock(LOCK, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    result = run_fixture_tracer(verified, FIXTURE, EXPECTED, case_name="perfect_linear")

    assert result["call_chain"] == ["evaluate", "per_sample_metrics", "summarise"]
    assert result["source_parity"] == "verified"
    assert result["private_container_parity"] == "unproven"
    assert result["official_counts"] == {
        "edge_tp": 2,
        "edge_fp": 0,
        "edge_fn": 0,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 0,
        "num_pred_nodes": 3,
    }
    assert result["official_summary"]["score"] == "1"


def test_lock_records_fixture_result_identity() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["cases"]["perfect_linear"]
    assert lock["fixtures"]["perfect_linear_result_sha256"] == sha256_bytes(
        canonical_json_bytes(expected)
    )


def test_lock_rejects_missing_checkout(tmp_path: Path) -> None:
    with pytest.raises(ScorerVerificationError, match="checkout_missing") as exc:
        verify_scorer_lock(LOCK, tmp_path / "missing", tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "checkout_missing"


def test_lock_rejects_changed_source_bytes(tmp_path: Path) -> None:
    copied = tmp_path / "organizer"
    shutil.copytree(CHECKOUT, copied)
    metrics = copied / "src" / "tracking_cellmot" / "metrics.py"
    metrics.write_bytes(metrics.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ScorerVerificationError, match="source_hash_mismatch") as exc:
        verify_scorer_lock(LOCK, copied, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "source_hash_mismatch"


def test_lock_rejects_wrong_commit(tmp_path: Path) -> None:
    copied = tmp_path / "organizer"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(CHECKOUT), str(copied)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(copied), "checkout", "--quiet", "HEAD^"],
        check=True,
    )
    with pytest.raises(ScorerVerificationError, match="checkout_commit_mismatch") as exc:
        verify_scorer_lock(LOCK, copied, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "checkout_commit_mismatch"


def test_lock_rejects_missing_dependency(tmp_path: Path) -> None:
    lock = _write_lock(
        tmp_path,
        lambda value: value["environment"]["packages"].__setitem__(
            "definitely-not-a-real-package", "1.0.0"
        ),
    )
    with pytest.raises(ScorerVerificationError, match="dependency_missing") as exc:
        verify_scorer_lock(lock, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "dependency_missing"


def test_lock_rejects_dependency_version_drift(tmp_path: Path) -> None:
    lock = _write_lock(
        tmp_path,
        lambda value: value["environment"]["packages"].__setitem__("polars", "0.0.0"),
    )
    with pytest.raises(ScorerVerificationError, match="dependency_version_mismatch") as exc:
        verify_scorer_lock(lock, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "dependency_version_mismatch"


def test_lock_rejects_import_path_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = tmp_path / "fake"
    package = fake_root / "tracking_cellmot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "metrics.py").write_text(
        "def evaluate(*a, **k): pass\n"
        "def per_sample_metrics(*a, **k): pass\n"
        "def summarise(*a, **k): pass\n"
        "def node_recall(*a, **k): pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(fake_root))
    for name in tuple(sys.modules):
        if name == "tracking_cellmot" or name.startswith("tracking_cellmot."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(ScorerVerificationError, match="import_path_escape") as exc:
        verify_scorer_lock(LOCK, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "import_path_escape"


def test_base_help_does_not_import_scientific_stack() -> None:
    code = (
        "import json, sys\n"
        "from biohub_tracker.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0\n"
        "scientific = {'numpy','scipy','polars','geff','tracksdata','torch','tracking_cellmot'}\n"
        "print(json.dumps(sorted(set(sys.modules) & scientific)))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.rstrip().endswith("[]")
