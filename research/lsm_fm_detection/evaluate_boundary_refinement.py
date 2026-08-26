#!/usr/bin/env python
"""Clean boundary sweep around the near-gate radius-2 feature-36 centroid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from evaluate_localization_refinement import (
        ACCEPTANCE_STEMS,
        PROMOTION_WORST_DELTA_MIN,
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        SCREEN_STEMS,
        Strategy,
        evaluate_movies,
        select_global_strategy,
        sha256_file,
        summarize_rows,
    )
except ModuleNotFoundError:
    from research.lsm_fm_detection.evaluate_localization_refinement import (
        ACCEPTANCE_STEMS,
        PROMOTION_WORST_DELTA_MIN,
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        SCREEN_STEMS,
        Strategy,
        evaluate_movies,
        select_global_strategy,
        sha256_file,
        summarize_rows,
    )


CONTROL_STRATEGY = "probability_r2_p2"
BOUNDARY_STRATEGIES = (
    Strategy(CONTROL_STRATEGY, "weighted", 2, 2.0, 0.0),
    Strategy("probability_r2_p1", "weighted", 2, 1.0, 0.0),
    Strategy("probability_r2_p1_5", "weighted", 2, 1.5, 0.0),
    Strategy("probability_r3_p1", "weighted", 3, 1.0, 0.0),
    Strategy("probability_r3_p1_5", "weighted", 3, 1.5, 0.0),
    Strategy("probability_r3_p2", "weighted", 3, 2.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("boundary-refinement evaluation requires CUDA")
    training_result = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training_result.get("status") != "completed":
        raise ValueError("training result is not complete")
    if training_result.get("validation_overlap") != []:
        raise ValueError("training result has validation overlap")
    if training_result.get("public_predictions_copied") is not False:
        raise ValueError("training result does not prove independent output")
    if sha256_file(args.model_path) != training_result.get("best_weight_sha256"):
        raise ValueError("learned checkpoint hash mismatch")

    try:
        from lsm_fm_model import EXPECTED_DETECTOR_PARAMETERS, build_lsm_fm_detector
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import (
            EXPECTED_DETECTOR_PARAMETERS,
            build_lsm_fm_detector,
        )

    detector = build_lsm_fm_detector(
        args.lsm_fm_checkpoint,
        expected_sha256=args.lsm_fm_checkpoint_sha256,
    )
    detector.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)["state_dict"],
        strict=True,
    )
    if sum(parameter.numel() for parameter in detector.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected detector parameter count")
    device = torch.device("cuda")
    detector.requires_grad_(False).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    selection_rows = evaluate_movies(
        detector,
        args.competition_dir,
        SCREEN_STEMS,
        device=device,
        batch_size=args.batch_size,
        calibration_frames=args.calibration_frames,
        strategies=BOUNDARY_STRATEGIES,
        baseline_predictions=None,
        max_wall_seconds=args.max_wall_seconds,
        started=started,
        partial_path=args.output_dir / "selection_boundary_partial.json",
    )
    selection = {name: summarize_rows(rows) for name, rows in selection_rows.items()}
    selected, diagnostics = select_global_strategy(
        selection, control_name=CONTROL_STRATEGY
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "selection": selection,
        "selection_diagnostics": diagnostics,
        "selected_strategy": selected,
        "selection_passed": selected is not None,
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.model_path),
            "training_result_sha256": sha256_file(args.training_result),
            "parameter_count": EXPECTED_DETECTOR_PARAMETERS,
            "control": CONTROL_STRATEGY,
            "strategy_scope": (
                "one global probability-centroid boundary strategy; peak identities, "
                "confidence ranking, and density calibration unchanged"
            ),
        },
    }
    if selected is not None:
        strategy = next(value for value in BOUNDARY_STRATEGIES if value.name == selected)
        acceptance_rows = evaluate_movies(
            detector,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            device=device,
            batch_size=args.batch_size,
            calibration_frames=args.calibration_frames,
            strategies=(strategy,),
            baseline_predictions=args.baseline_predictions,
            max_wall_seconds=args.max_wall_seconds,
            started=started,
            partial_path=args.output_dir / "acceptance_boundary_partial.json",
        )[selected]
        acceptance = summarize_rows(acceptance_rows)
        deltas = [float(row["annotated_recall_delta"]) for row in acceptance_rows]
        prefix_pass = all(
            acceptance["by_prefix"][prefix]["annotated_node_recall"]
            >= PUBLIC_PREFIX_RECALL[prefix] - 0.01
            for prefix in PUBLIC_PREFIX_RECALL
        )
        result.update(
            {
                "acceptance": acceptance,
                "acceptance_opened": True,
                "promotion_passed": (
                    acceptance["annotated_node_recall"] >= PUBLIC_ACCEPTANCE_RECALL
                    and min(deltas) >= PROMOTION_WORST_DELTA_MIN
                    and prefix_pass
                ),
            }
        )
    output = args.output_dir / "lsm_fm_boundary_refinement.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LSM-FM BOUNDARY COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
