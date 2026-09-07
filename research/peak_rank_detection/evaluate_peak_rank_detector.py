#!/usr/bin/env python
"""Two-GPU clean Kaggle validation for the temporal peak-ranking detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
    from evaluate_pretrained_detector import (
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )
    from inference import predict_frames
    from model import TemporalPeakRankDetector, count_parameters
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices
    from research.peak_rank_detection.inference import predict_frames
    from research.peak_rank_detection.model import (
        TemporalPeakRankDetector,
        count_parameters,
    )
    from research.spotiflow_biohub.evaluate_pretrained_detector import (
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )


SCREEN_STEMS = (
    "44b6_d29c9ab2",
    "44b6_3a861e03",
    "44b6_d5e7d891",
    "44b6_ddf577ad",
    "6bba_09961292",
    "6bba_bb9f20c3",
    "6bba_784a78c9",
    "6bba_57b7cc1e",
)
ACCEPTANCE_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
PUBLIC_ACCEPTANCE_RECALL = 0.96902842596521
PUBLIC_PREFIX_RECALL = {"44b6": 0.9588014981273408, "6bba": 0.9775019394879751}
SELECTION_POOLED_MIN = 0.80
SELECTION_WORST_MIN = 0.65
PROMOTION_WORST_DELTA_MIN = -0.01
EXPECTED_PARAMETER_COUNT = 38_381_478


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_training(checkpoint: Path, terminal_path: Path) -> dict[str, Any]:
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("status") == "accepted_at_audit"
        and terminal.get("selection_passed") is True
        and terminal.get("audit_opened") is True
        and terminal.get("audit_passed") is True
        and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_predictions_read") is False
        and terminal.get("public_notebook_weights_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("training terminal is not an accepted clean checkpoint")
    if sha256_file(checkpoint) != terminal.get("checkpoint_sha256"):
        raise ValueError("detector checkpoint hash differs from training terminal")
    return terminal


def load_model(
    checkpoint: Path, terminal: dict[str, Any], device: torch.device
) -> TemporalPeakRankDetector:
    model = TemporalPeakRankDetector(
        widths=terminal["widths"], depths=terminal["depths"]
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if count_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("unexpected detector parameter count")
    return model.requires_grad_(False).eval().to(device)


def evaluate_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    terminal = validate_training(args.checkpoint, args.training_terminal)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in detector validation worker")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, terminal, device)
    import zarr

    rows = []
    for stem in args.stems.split(","):
        if not stem:
            continue
        sample_path = args.competition_dir / "train" / f"{stem}.zarr"
        truth_path = args.competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing organizer count estimate for {stem}")
        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        predictions, observed_frames = predict_frames(
            model,
            sample_path,
            range(frame_count),
            device=device,
            batch_size=args.batch_size,
            d4_tta=not args.disable_d4_tta,
        )
        if observed_frames != frame_count:
            raise RuntimeError("detector inference frame count changed")
        sampled = set(uniform_frame_indices(frame_count, args.calibration_frames).tolist())
        threshold, projected = density_threshold(
            [row for row in predictions if row.frame in sampled], estimated, frame_count
        )
        # Labels are first opened after the image/metadata-only threshold is fixed.
        truth = graph_points_by_frame(truth_path)
        candidate = score_predictions(predictions, truth, threshold)
        row: dict[str, Any] = {
            "stem": stem,
            "estimated_node_count": estimated,
            "threshold": threshold,
            "projected_node_count": projected,
            "projected_count_ratio": projected / estimated,
            "candidate": candidate,
        }
        if args.baseline_predictions is not None:
            baseline = score_graph_nodes(
                graph_points_by_frame(args.baseline_predictions / f"{stem}.geff"),
                truth,
                frame_count,
            )
            row["baseline_raw_graph"] = baseline
            row["annotated_recall_delta"] = float(
                candidate["annotated_node_recall"]
            ) - float(baseline["annotated_node_recall"])
        rows.append(row)
        print("PEAK RANK EVAL", json.dumps(row, sort_keys=True), flush=True)
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("peak-ranking validation worker reached its wall guard")
    atomic_json(args.output, {"rows": rows})


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in rows)
    by_prefix = {}
    for prefix in ("44b6", "6bba"):
        selected = [row for row in rows if str(row["stem"]).startswith(prefix)]
        prefix_annotated = sum(
            int(row["candidate"]["annotated_gt_nodes"]) for row in selected
        )
        prefix_matched = sum(
            int(row["candidate"]["matched_gt_nodes"]) for row in selected
        )
        by_prefix[prefix] = {
            "annotated_gt_nodes": prefix_annotated,
            "matched_gt_nodes": prefix_matched,
            "annotated_node_recall": prefix_matched / prefix_annotated,
        }
    return {
        "movies": len(rows),
        "annotated_gt_nodes": annotated,
        "matched_gt_nodes": matched,
        "annotated_node_recall": matched / annotated,
        "worst_movie_recall": min(
            float(row["candidate"]["annotated_node_recall"]) for row in rows
        ),
        "by_prefix": by_prefix,
        "rows": list(rows),
    }


def launch_workers(
    args: argparse.Namespace,
    stems: Sequence[str],
    *,
    phase: str,
    use_baseline: bool,
) -> list[dict[str, Any]]:
    devices = [value for value in args.devices.split(",") if value]
    if len(devices) != 2:
        raise ValueError("exactly two CUDA device identifiers are required")
    chunks = [tuple(stems[index::2]) for index in range(2)]
    processes = []
    outputs = []
    for index, (device, chunk) in enumerate(zip(devices, chunks, strict=True)):
        output = args.output_dir / f"{phase}-worker-{index}.json"
        outputs.append(output)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--competition-dir",
            str(args.competition_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--training-terminal",
            str(args.training_terminal),
            "--stems",
            ",".join(chunk),
            "--device",
            "cuda:0",
            "--batch-size",
            str(args.batch_size),
            "--calibration-frames",
            str(args.calibration_frames),
            "--max-wall-seconds",
            str(args.max_wall_seconds),
            "--output",
            str(output),
        ]
        if args.disable_d4_tta:
            command.append("--disable-d4-tta")
        if use_baseline:
            command.extend(["--baseline-predictions", str(args.baseline_predictions)])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        processes.append(subprocess.Popen(command, env=environment))
    exit_codes = [process.wait() for process in processes]
    if exit_codes != [0, 0]:
        raise RuntimeError(f"{phase} validation workers failed: {exit_codes}")
    by_stem = {}
    for output in outputs:
        for row in json.loads(output.read_text(encoding="utf-8"))["rows"]:
            if row["stem"] in by_stem:
                raise ValueError(f"duplicate worker result for {row['stem']}")
            by_stem[row["stem"]] = row
    if set(by_stem) != set(stems):
        raise ValueError(f"{phase} worker stem inventory changed")
    return [by_stem[stem] for stem in stems]


def orchestrate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    terminal = validate_training(args.checkpoint, args.training_terminal)
    selection_rows = launch_workers(
        args, SCREEN_STEMS, phase="selection", use_baseline=False
    )
    selection = summarize(selection_rows)
    selected = bool(
        selection["annotated_node_recall"] >= SELECTION_POOLED_MIN
        and selection["worst_movie_recall"] >= SELECTION_WORST_MIN
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "temporal-peak-rank-clean-validation-v1",
        "status": "completed",
        "selection": selection,
        "selection_passed": selected,
        "selection_gate": {
            "pooled_recall_min": SELECTION_POOLED_MIN,
            "worst_movie_recall_min": SELECTION_WORST_MIN,
        },
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "training_terminal_sha256": sha256_file(args.training_terminal),
            "parameter_count": terminal["parameter_count"],
            "two_gpu_workers": True,
            "d4_tta": not args.disable_d4_tta,
        },
    }
    if selected:
        acceptance_rows = launch_workers(
            args, ACCEPTANCE_STEMS, phase="acceptance", use_baseline=True
        )
        acceptance = summarize(acceptance_rows)
        deltas = [float(row["annotated_recall_delta"]) for row in acceptance_rows]
        prefix_pass = all(
            acceptance["by_prefix"][prefix]["annotated_node_recall"]
            >= PUBLIC_PREFIX_RECALL[prefix] - 0.01
            for prefix in PUBLIC_PREFIX_RECALL
        )
        promoted = bool(
            acceptance["annotated_node_recall"] >= PUBLIC_ACCEPTANCE_RECALL
            and min(deltas) >= PROMOTION_WORST_DELTA_MIN
            and prefix_pass
        )
        result.update(
            {
                "acceptance": acceptance,
                "acceptance_opened": True,
                "promotion_passed": promoted,
                "promotion_gate": {
                    "public_pooled_recall": PUBLIC_ACCEPTANCE_RECALL,
                    "public_prefix_recall": PUBLIC_PREFIX_RECALL,
                    "prefix_regression_tolerance": 0.01,
                    "worst_movie_delta_min": PROMOTION_WORST_DELTA_MIN,
                },
            }
        )
    unexpected = [
        path
        for path in args.output_dir.parent.rglob("*")
        if path.is_file() and path.name.lower() in {"submission.csv", "submission.zip"}
    ]
    if unexpected:
        raise RuntimeError(f"validation created forbidden submission files: {unexpected}")
    atomic_json(args.output_dir / "peak_rank_validation.json", result)
    print("PEAK RANK VALIDATION COMPLETE", json.dumps(result, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-terminal", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--stems", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=39_000.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-d4-tta", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.calibration_frames <= 0 or args.max_wall_seconds <= 0:
        parser.error("batch size, calibration frames, and wall limit must be positive")
    if args.worker and args.output is None:
        parser.error("workers require --output")
    if not args.worker and (args.output_dir is None or args.baseline_predictions is None):
        parser.error("orchestration requires --output-dir and --baseline-predictions")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        evaluate_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
