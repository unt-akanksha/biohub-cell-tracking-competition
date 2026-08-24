from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from decimal import Decimal
from pathlib import Path

import pytest

from biohub_tracker.cli import build_parser
from biohub_tracker.io import canonical_json_bytes, sha256_bytes
from biohub_tracker.scorer import run_fixture_tracer, score_fixture_case, score_fixture_set
from biohub_tracker.scorer_lock import ScorerVerificationError, verify_scorer_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "official-scorer.lock.json"
CHECKOUT = ROOT / ".biohub" / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA = ROOT / ".biohub" / "vendor" / "tracksdata"
FIXTURE = ROOT / "tests" / "fixtures" / "metric" / "graph_specs" / "perfect-linear.json"
EXPECTED = ROOT / "tests" / "fixtures" / "metric" / "expected" / "official-counts.json"
REGRESSIONS = ROOT / "tests" / "fixtures" / "metric" / "graph_specs" / "metric-regressions.json"
EXPLOIT = ROOT / "tests" / "fixtures" / "metric" / "graph_specs" / "patched-exploit.json"


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
    for source in (FIXTURE, EXPLOIT):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
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


@pytest.fixture(scope="module")
def scorer_and_regressions():
    verified = verify_scorer_lock(LOCK, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    fixture = json.loads(REGRESSIONS.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["cases"]
    return verified, fixture, expected


@pytest.mark.parametrize(
    "case_name",
    [
        "boundary_exact_7um",
        "boundary_over_7um",
        "time_mismatch",
        "anisotropic_z_reject",
        "missing_node",
        "sparse_boundary_ignored",
        "sparse_interior_penalty",
    ],
)
def test_edge_matching_regressions(scorer_and_regressions, case_name: str) -> None:
    verified, fixture, expected = scorer_and_regressions
    assert score_fixture_case(verified, fixture["cases"][case_name]) == expected[case_name]


@pytest.mark.parametrize("case_name", ["penalty_equal", "penalty_over", "penalty_under"])
def test_penalty_regressions(scorer_and_regressions, case_name: str) -> None:
    verified, fixture, expected = scorer_and_regressions
    result = score_fixture_case(verified, fixture["cases"][case_name])
    assert result == expected[case_name]
    if case_name == "penalty_under":
        assert Decimal(result["official_summary"]["adj_edge_jaccard"]) > Decimal("1")


@pytest.mark.parametrize("case_name", ["division_ontime", "division_early", "division_late"])
def test_division_regressions(scorer_and_regressions, case_name: str) -> None:
    verified, fixture, expected = scorer_and_regressions
    result = score_fixture_case(verified, fixture["cases"][case_name])
    assert result == expected[case_name]
    assert result["official_counts"]["division_tp"] == 1


def test_aggregate_uses_official_weighted_micro_summary(scorer_and_regressions) -> None:
    verified, fixture, expected = scorer_and_regressions
    names = fixture["aggregation_sets"]["imbalanced"]
    result = score_fixture_set(verified, [(name, fixture["cases"][name]) for name in names])
    for key, value in expected["aggregate_imbalanced"].items():
        assert result[key] == value
    assert result["official_summary"]["score"] != result["arithmetic_movie_score"]
    assert result["adjusted_edge_inputs"]["weights"] == [5, 2]


def test_aggregate_no_division_drops_term_without_nan(scorer_and_regressions) -> None:
    verified, fixture, expected = scorer_and_regressions
    names = fixture["aggregation_sets"]["no_divisions"]
    result = score_fixture_set(verified, [(name, fixture["cases"][name]) for name in names])
    for key, value in expected["aggregate_no_divisions"].items():
        assert result[key] == value
    assert result["official_summary"]["division_jaccard"] is None
    assert result["official_summary"]["score"] is not None


def test_edge_fixture_evaluation_uses_fresh_copies(scorer_and_regressions) -> None:
    verified, fixture, _ = scorer_and_regressions
    case = fixture["cases"]["division_ontime"]
    before = canonical_json_bytes(case)
    first = score_fixture_case(verified, case)
    second = score_fixture_case(verified, case)
    assert first == second
    assert canonical_json_bytes(case) == before


def _graph_with_named_ids(verified, spec):
    import polars as pl

    graph = verified.tracksdata.graph.InMemoryGraph()
    for coordinate in ("z", "y", "x"):
        graph.add_node_attr_key(coordinate, pl.Float64, 0.0)
    ids = {
        node["id"]: graph.add_node(
            {key: node[key] for key in ("t", "z", "y", "x")}
        )
        for node in spec["nodes"]
    }
    for source, target in spec["edges"]:
        graph.add_edge(ids[source], ids[target], {})
    return graph, ids


@pytest.fixture(scope="module")
def scorer_and_exploit():
    verified = verify_scorer_lock(LOCK, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    fixture = json.loads(EXPLOIT.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["cases"]
    return verified, fixture, expected


def test_hack2_exploit_exact_patched_counts_and_forks(scorer_and_exploit) -> None:
    verified, fixture, expected = scorer_and_exploit
    case = fixture["cases"]["hack2"]
    result = score_fixture_case(verified, case)
    assert result == expected["hack2"]
    assert result["official_counts"] == {
        "edge_tp": 5,
        "edge_fp": 4,
        "edge_fn": 5,
        "division_tp": 0,
        "division_fp": 4,
        "division_fn": 2,
        "num_pred_nodes": 15,
    }
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["fixtures"]["hack2_result_sha256"] == sha256_bytes(
        canonical_json_bytes(expected["hack2"])
    )

    pred, pred_ids = _graph_with_named_ids(verified, case["prediction"])
    truth, _ = _graph_with_named_ids(verified, case["truth"])
    from tracking_cellmot.division_metrics import score_divisions

    forks = score_divisions(pred, truth, max_distance=case["max_distance"])
    fp_names = {name for name, node_id in pred_ids.items() if node_id in forks.fp_forks}
    assert fp_names == {"105", "106", "122", "124"}


@pytest.mark.parametrize(
    "case_name",
    ["fork_reuse", "cross_component_fork", "merged_shared_grandchild", "malformed_same_lineage"],
)
def test_malformed_fork_regressions(scorer_and_exploit, case_name: str) -> None:
    verified, fixture, expected = scorer_and_exploit
    case = fixture["cases"][case_name]
    result = score_fixture_case(verified, case)
    assert result == expected[case_name]

    pred, pred_ids = _graph_with_named_ids(verified, case["prediction"])
    truth, _ = _graph_with_named_ids(verified, case["truth"])
    from tracking_cellmot.division_metrics import score_divisions

    forks = score_divisions(pred, truth, max_distance=case["max_distance"])
    fp_names = {name for name, node_id in pred_ids.items() if node_id in forks.fp_forks}
    assert fp_names == set(fixture["fork_expectations"][case_name])
    if case_name == "fork_reuse":
        assert sum(forks.scores.values()) == 1
        assert len(forks.scores) == 2


def test_exploit_patch_identity_change_fails_before_fixture(tmp_path: Path) -> None:
    lock = _write_lock(
        tmp_path,
        lambda value: value["organizer"].__setitem__("patch_commit", "0" * 40),
    )
    with pytest.raises(ScorerVerificationError, match="patch_commit_missing") as exc:
        verify_scorer_lock(lock, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "patch_commit_missing"


def test_exploit_fixture_hash_change_fails_before_scoring(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, lambda value: None)
    fixture = tmp_path / EXPLOIT.relative_to(ROOT)
    fixture.write_bytes(fixture.read_bytes() + b"\n")
    with pytest.raises(ScorerVerificationError, match="fixture_hash_mismatch") as exc:
        verify_scorer_lock(lock, CHECKOUT, tracksdata_checkout=TRACKSDATA)
    assert exc.value.reason_code == "fixture_hash_mismatch"


def test_exploit_old_scorer_is_not_production_reachable() -> None:
    import inspect

    parser_help = build_parser().format_help()
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "src" / "biohub_tracker" / "cli.py",
            ROOT / "src" / "biohub_tracker" / "scorer.py",
            ROOT / "src" / "biohub_tracker" / "scorer_lock.py",
        )
    )
    assert "--scorer-version" not in parser_help
    assert "7396b7e98e61844e799152ddda7e5493084cc8f3" not in production_text
    assert "prepatch" not in production_text.lower()
    assert "version" not in inspect.signature(verify_scorer_lock).parameters


def test_exploit_fixture_distinguishes_isolated_prepatch(tmp_path: Path) -> None:
    fixture = json.loads(EXPLOIT.read_text(encoding="utf-8"))
    old_checkout = tmp_path / "isolated-old-source"
    archive = tmp_path / "isolated-old-source.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(CHECKOUT),
            "archive",
            "--format=tar",
            f"--output={archive}",
            fixture["prepatch_commit"],
            "src/tracking_cellmot",
        ],
        check=True,
    )
    with tarfile.open(archive) as handle:
        handle.extractall(old_checkout, filter="data")
    code = """
import json, sys
from pathlib import Path
import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import evaluate

case = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['cases']['hack2']
def build(spec):
    graph = td.graph.InMemoryGraph()
    for coordinate in ('z', 'y', 'x'):
        graph.add_node_attr_key(coordinate, pl.Float64, 0.0)
    ids = {node['id']: graph.add_node({key: node[key] for key in ('t','z','y','x')}) for node in spec['nodes']}
    for source, target in spec['edges']:
        graph.add_edge(ids[source], ids[target], {})
    return graph
result = evaluate(build(case['prediction']), build(case['truth']), max_distance=case['max_distance'])
print(json.dumps(result._asdict(), sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(old_checkout / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(EXPLOIT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    old_counts = json.loads(completed.stdout.splitlines()[-1])
    patched = json.loads(EXPECTED.read_text(encoding="utf-8"))["cases"]["hack2"][
        "official_counts"
    ]
    assert old_counts != patched
