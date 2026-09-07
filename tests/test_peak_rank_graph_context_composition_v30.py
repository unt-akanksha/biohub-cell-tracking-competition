from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-peak-rank-graph-context-composition-v30.py"
VERIFIER = ROOT / "scripts/verify-peak-rank-graph-context-composition-v30.py"
CONTROLLER = ROOT / "scripts/wait-build-verify-submit-peak-graph-composition-v30.ps1"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_composes_graph_after_production_and_around_validator() -> None:
    module = load(BUILDER, "v30_builder")
    notebook = module.build_notebook("a" * 64, "b" * 64)
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

    inference = next(i for i, source in enumerate(cells) if "Prediction completed in" in source)
    graph_setup = next(i for i, source in enumerate(cells) if "_GCD_MANIFEST_SHA256" in source)
    post = next(i for i, source in enumerate(cells) if "def motion_relink_edges(" in source)
    offload = next(i for i, source in enumerate(cells) if "parked on CPU" in source)
    validator = next(i for i, source in enumerate(cells) if "predict_val_cmd = [" in source)
    restore = next(i for i, source in enumerate(cells) if "restored for complete-movie" in source)
    evidence = next(i for i, source in enumerate(cells) if "graph_ranked_edges_added" in source)

    assert inference < graph_setup < post < offload < validator < restore < evidence
    assert cells[post].count("def _graph_context_scores_for_candidates") == 1
    assert cells[post].count("edges, ranked_consensus_stats =") == 1
    assert "ranked consensus produced a nonconsecutive edge" in cells[post]
    for cell, source in zip(notebook["cells"], cells, strict=True):
        if cell.get("cell_type") == "code":
            ast.parse(source)
    assert notebook["metadata"]["codex"]["component_order"] == [
        "peak_rank_detector_v28",
        "graph_context_division_v2",
    ]
    all_code = "\n".join(cells)
    assert "kaggle competitions submit" not in all_code


def test_verifier_requires_exact_gain_over_v28_and_graph_policy_unit() -> None:
    module = load(VERIFIER, "v30_verifier")
    source = VERIFIER.read_text(encoding="utf-8")
    scores = module.candidate_scores_by_movie(
        {
            "official_metric_result": {
                "by_movie": [
                    {"sample_id": f"movie-{index}", "candidate": {"score": index / 10}}
                    for index in range(4)
                ]
            }
        }
    )

    assert scores["movie-3"] == 0.3
    assert module.MINIMUM_COMPOSITION_GAIN > 0
    assert module.MAXIMUM_MOVIE_REGRESSION == 0.001
    assert 'graph.get("policy_unit_audited") is True' in source
    assert 'prior.get("run_id") == PEAK_RUN_ID' in source
    assert "gain >= MINIMUM_COMPOSITION_GAIN" in source
    assert "worst_movie_delta >= -MAXIMUM_MOVIE_REGRESSION" in source


def test_controller_preserves_quota_and_patched_official_gates() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert '$gpuReserveHours = 8.0' in source
    assert '$declaredWorstCaseGpuHours = 12.0' in source
    assert '"Global\\BiohubKaggleGpuSessionV1"' in source
    assert "score_official_candidate.py" in source
    assert "--peak-promotion" in source
    assert "composition_gain_over_peak_component" in source
    assert "& kaggle competitions submit" not in source
