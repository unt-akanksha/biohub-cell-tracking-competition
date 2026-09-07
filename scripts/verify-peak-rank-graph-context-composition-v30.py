#!/usr/bin/env python3
"""Promote V30 only when it beats its independently promoted V28 detector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
PEAK = runpy.run_path(str(ROOT / "scripts/verify-peak-rank-submission-candidate.py"))
GRAPH = runpy.run_path(
    str(ROOT / "scripts/verify-graph-context-consensus-submission-candidate.py")
)
RUN_ID = "peak-rank-graph-context-composition-v30"
PEAK_RUN_ID = "peak-rank-xl-hard-mined-temporal-snr-pair-tracking-candidate-v28"
GRAPH_POLICY_CONTRACT = "all-selection-admitted-equal-rank-ensemble-v2"
MINIMUM_COMPOSITION_GAIN = 1e-6
MAXIMUM_MOVIE_REGRESSION = 0.001


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_scores_by_movie(result: dict) -> dict[str, float]:
    rows = result.get("official_metric_result", {}).get("by_movie", [])
    scores = {
        str(row["sample_id"]): float(row["candidate"]["score"])
        for row in rows
    }
    if len(scores) != 4 or len(rows) != 4 or not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError("composition component per-movie evidence is invalid")
    return scores


def verify(
    *,
    output_root: Path,
    baseline_validator: Path,
    peak_runtime_manifest: Path,
    graph_runtime_manifest: Path,
    graph_development_terminal: Path,
    peak_promotion: Path,
    official_metric_result: Path,
) -> dict:
    peak = PEAK["verify_candidate"](
        output_root,
        baseline_validator,
        peak_runtime_manifest,
        official_metric_result,
        expected_run_id=RUN_ID,
    )
    graph = GRAPH["validate_runtime"](graph_runtime_manifest)
    development = json.loads(graph_development_terminal.read_text(encoding="utf-8"))
    prior = json.loads(peak_promotion.read_text(encoding="utf-8"))
    evidence_path = PEAK["unique_file"](output_root, "candidate_evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not (
        graph.get("deep_policy") == "equal_rank_selection_admitted_ensemble"
        and graph.get("deep_member_count", 0) >= 2
        and graph.get("policy_contract") == GRAPH_POLICY_CONTRACT
        and graph.get("policy_unit_audited") is True
        and graph.get("constituent_audit_gate_required") is False
        and development.get("status") == "runtime_packaged"
        and development.get("runtime_created") is True
        and development.get("authorized_for_full_candidate_evaluation") is True
        and development.get("authorized_for_submission") is False
        and development.get("runtime_manifest_sha256")
        == GRAPH["sha256_file"](graph_runtime_manifest)
        and prior.get("status") == "eligible_for_submission"
        and prior.get("run_id") == PEAK_RUN_ID
        and prior.get("authorized_for_submission") is True
        and prior.get("competition_submission_performed") is False
        and prior.get("runtime_manifest_sha256")
        == PEAK["sha256_file"](peak_runtime_manifest)
        and evidence.get("component_order")
        == ["peak_rank_detector_v28", "graph_context_division_v2"]
        and evidence.get("independent_component_promotion_required") is True
        and evidence.get("graph_runtime_manifest_sha256")
        == graph["manifest_sha256"]
        and evidence.get("graph_policy_sha256") == graph["policy_sha256"]
        and evidence.get("graph_policy") == graph["deep_policy"]
        and evidence.get("graph_policy_contract") == GRAPH_POLICY_CONTRACT
        and evidence.get("graph_policy_unit_audited") is True
        and evidence.get("graph_member_count") == graph["deep_member_count"]
        and evidence.get("graph_parameter_count_per_member") == 74_732_308
        and isinstance(evidence.get("graph_ranked_candidates"), int)
        and isinstance(evidence.get("graph_ranked_agreements"), int)
        and isinstance(evidence.get("graph_ranked_edges_added"), int)
        and 0 <= evidence["graph_ranked_edges_added"] <= evidence["graph_ranked_agreements"]
        <= evidence["graph_ranked_candidates"]
    ):
        raise RuntimeError("V30 independent component or evidence contract failed")

    prior_score = float(prior.get("exact_candidate_score", math.nan))
    composition_score = float(peak["exact_candidate_score"])
    gain = composition_score - prior_score
    current_movies = candidate_scores_by_movie(peak)
    prior_movies = candidate_scores_by_movie(prior)
    if set(current_movies) != set(prior_movies):
        raise RuntimeError("V30 and V28 complete-movie inventories differ")
    movie_deltas = {
        stem: current_movies[stem] - prior_movies[stem] for stem in sorted(current_movies)
    }
    worst_movie_delta = min(movie_deltas.values())
    if not (
        math.isfinite(gain)
        and gain >= MINIMUM_COMPOSITION_GAIN
        and worst_movie_delta >= -MAXIMUM_MOVIE_REGRESSION
    ):
        raise RuntimeError(
            "V30 did not improve the independently promoted V28 component: "
            f"gain={gain:.9f}, worst_movie_delta={worst_movie_delta:.9f}"
        )
    return {
        **peak,
        "run_id": RUN_ID,
        "component_order": ["peak_rank_detector_v28", "graph_context_division_v2"],
        "independent_component_promotion_required": True,
        "peak_component_promotion_sha256": PEAK["sha256_file"](peak_promotion),
        "graph_development_terminal_sha256": GRAPH["sha256_file"](
            graph_development_terminal
        ),
        "graph_runtime_manifest_sha256": graph["manifest_sha256"],
        "graph_policy_sha256": graph["policy_sha256"],
        "graph_member_count": graph["deep_member_count"],
        "composition_gain_over_peak_component": gain,
        "composition_movie_deltas": movie_deltas,
        "composition_worst_movie_delta": worst_movie_delta,
        "authorized_for_submission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validator", type=Path, required=True)
    parser.add_argument("--peak-runtime-manifest", type=Path, required=True)
    parser.add_argument("--graph-runtime-manifest", type=Path, required=True)
    parser.add_argument("--graph-development-terminal", type=Path, required=True)
    parser.add_argument("--peak-promotion", type=Path, required=True)
    parser.add_argument("--official-metric-result", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(
        output_root=args.output_root,
        baseline_validator=args.baseline_validator,
        peak_runtime_manifest=args.peak_runtime_manifest,
        graph_runtime_manifest=args.graph_runtime_manifest,
        graph_development_terminal=args.graph_development_terminal,
        peak_promotion=args.peak_promotion,
        official_metric_result=args.official_metric_result,
    )
    if args.report is not None:
        atomic_json(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
