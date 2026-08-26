from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from multibackbone import (
        build_multibackbone_variants,
        materialize_variant,
        required_backbones,
    )
    from train_biohub_hoct_probe import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        GraphVideo,
        ProbeConfig,
        _score_partition,
        _selection_configurations,
        atomic_json,
        extract_training_examples,
        fit_probe,
        link_configuration,
        predict_movie_variants,
        read_graph_video,
        read_raw_graphs,
        read_submission,
        sha256_file,
        summarize_stored_edge_probabilities,
        transfer_raw_edge_probabilities,
    )
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.hoct_graph.multibackbone import (
        build_multibackbone_variants,
        materialize_variant,
        required_backbones,
    )
    from research.hoct_graph.train_biohub_hoct_probe import (
        ACCEPTANCE_STEMS,
        SELECTION_STEMS,
        VALIDATION_STEMS,
        GraphVideo,
        ProbeConfig,
        _score_partition,
        _selection_configurations,
        atomic_json,
        extract_training_examples,
        fit_probe,
        link_configuration,
        predict_movie_variants,
        read_graph_video,
        read_raw_graphs,
        read_submission,
        sha256_file,
        summarize_stored_edge_probabilities,
        transfer_raw_edge_probabilities,
    )


def multibackbone_configurations(variant_names: list[str]) -> list[dict[str, Any]]:
    """Expand the v1 linker grid over a constrained list of model variants."""
    templates = [
        {key: value for key, value in row.items() if key != "variant"}
        for row in _selection_configurations()
        if row["variant"] == "pretrained"
    ]
    return [
        {"variant": variant, **template}
        for variant in sorted(variant_names)
        for template in templates
    ]


def validate_multibackbone_movies(
    models: dict[str, torch.nn.Module],
    probes: dict[str, torch.nn.Linear],
    predictions: dict[str, GraphVideo],
    train_dir: Path,
    device: torch.device,
    config: ProbeConfig,
    output_dir: Path,
    core_size: np.ndarray,
) -> dict[str, Any]:
    truths = {
        stem: read_graph_video(train_dir / f"{stem}.geff")
        for stem in VALIDATION_STEMS
    }
    selection_scores: dict[
        str, dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ] = {}
    for stem in SELECTION_STEMS:
        print(f"HOCT V2 CLEAN SELECTION INFERENCE: {stem}", flush=True)
        by_backbone = {
            name: predict_movie_variants(
                model,
                probes[name],
                predictions[stem],
                device,
                config,
                core_size=core_size,
            )
            for name, model in models.items()
        }
        selection_scores[stem] = build_multibackbone_variants(by_backbone)

    variant_names = sorted(next(iter(selection_scores.values())))
    if any(sorted(scores) != variant_names for scores in selection_scores.values()):
        raise RuntimeError("HOCT V2 selection movies produced inconsistent variants")
    base_edges = {
        stem: [tuple(map(int, edge)) for edge in predictions[stem].edges.tolist()]
        for stem in VALIDATION_STEMS
    }
    base_summary = {
        "selection": _score_partition(
            predictions, truths, train_dir, base_edges, SELECTION_STEMS
        ),
        "acceptance": _score_partition(
            predictions, truths, train_dir, base_edges, ACCEPTANCE_STEMS
        ),
        "all": _score_partition(
            predictions, truths, train_dir, base_edges, VALIDATION_STEMS
        ),
    }

    configurations: list[dict[str, Any]] = []
    for candidate in multibackbone_configurations(variant_names):
        edges = {
            stem: link_configuration(
                predictions[stem], selection_scores[stem][candidate["variant"]], candidate
            )
            for stem in SELECTION_STEMS
        }
        configurations.append(
            {
                **candidate,
                "selection_summary": _score_partition(
                    predictions, truths, train_dir, edges, SELECTION_STEMS
                ),
            }
        )
    selected = max(
        configurations,
        key=lambda row: (
            row["selection_summary"]["proxy_score"],
            row["selection_summary"]["worst_movie"],
            -row["selection_summary"]["div_fp"],
            row["method"] == "raw_confidence_hybrid",
        ),
    )
    selected_variant = str(selected["variant"])
    needed = required_backbones(selected_variant)

    # Only after the variant, linker, and thresholds are frozen do we read the
    # two acceptance movies. Single-backbone winners do not waste inference on
    # the unused model.
    acceptance_scores: dict[
        str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    for stem in ACCEPTANCE_STEMS:
        print(f"HOCT V2 CLEAN ACCEPTANCE INFERENCE: {stem}", flush=True)
        by_backbone = {
            name: predict_movie_variants(
                models[name],
                probes[name],
                predictions[stem],
                device,
                config,
                core_size=core_size,
            )
            for name in sorted(needed)
        }
        acceptance_scores[stem] = materialize_variant(
            selected_variant, by_backbone
        )

    selected_edges = {
        stem: link_configuration(
            predictions[stem],
            (
                selection_scores[stem][selected_variant]
                if stem in selection_scores
                else acceptance_scores[stem]
            ),
            selected,
        )
        for stem in VALIDATION_STEMS
    }
    acceptance_summary = _score_partition(
        predictions, truths, train_dir, selected_edges, ACCEPTANCE_STEMS
    )
    all_summary = _score_partition(
        predictions, truths, train_dir, selected_edges, VALIDATION_STEMS
    )
    selected = {
        **selected,
        "required_backbones": sorted(needed),
        "acceptance_summary": acceptance_summary,
        "all_summary": all_summary,
    }
    selection_delta = (
        selected["selection_summary"]["proxy_score"]
        - base_summary["selection"]["proxy_score"]
    )
    acceptance_delta = (
        acceptance_summary["proxy_score"]
        - base_summary["acceptance"]["proxy_score"]
    )
    acceptance_passed = bool(
        acceptance_delta > 0
        and acceptance_summary["worst_movie"]
        >= base_summary["acceptance"]["worst_movie"] - 0.01
    )
    payload = {
        "schema_version": 1,
        "model_family": "Higher-Order Cell Tracking Transformer multi-backbone",
        "backbones": sorted(models),
        "prediction_graph_kind": "postprocessed_public_comparator_with_transferred_raw_confidence",
        "validation_stems": list(VALIDATION_STEMS),
        "selection_stems": list(SELECTION_STEMS),
        "acceptance_stems": list(ACCEPTANCE_STEMS),
        "selection_rule": "backbone/head/blend/linker/thresholds selected on selection movies; acceptance inferred and scored once after freeze",
        "base_graph": base_summary,
        "selected": selected,
        "selection_delta_vs_base": selection_delta,
        "acceptance_delta_vs_base": acceptance_delta,
        "acceptance_passed": acceptance_passed,
        "stored_edge_probability_evidence": summarize_stored_edge_probabilities(
            predictions
        ),
        "configurations": configurations,
    }
    atomic_json(output_dir / "complete_movie_validation.json", payload)
    return payload


def _training_paths(train_dir: Path, train_per_prefix: int) -> list[Path]:
    excluded = set(VALIDATION_STEMS)
    by_prefix: dict[str, list[Path]] = {}
    for path in train_dir.glob("*.geff"):
        if path.stem not in excluded:
            by_prefix.setdefault(path.stem.split("_")[0], []).append(path)
    selected_paths: list[Path] = []
    for prefix, paths in sorted(by_prefix.items()):
        ranked = sorted(
            paths,
            key=lambda path: hashlib.sha256(path.stem.encode()).hexdigest(),
        )
        selected = ranked[:train_per_prefix]
        selected_paths.extend(selected)
        print(f"HOCT V2 TRAIN SELECT: {prefix} -> {len(selected)} movies", flush=True)
    if len(selected_paths) != 195:
        raise RuntimeError(
            f"Expected all 195 non-validation movies, found {len(selected_paths)}"
        )
    return selected_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--general-model", type=Path, required=True)
    parser.add_argument("--ctc-model", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--processed-validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-prefix", type=int, default=128)
    parser.add_argument("--core-size", type=float, default=128.0)
    parser.add_argument("--max-wall-seconds", type=float, default=13_200.0)
    args = parser.parse_args()

    started = time.monotonic()
    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for HOCT V2 feature extraction")
    torch.backends.cuda.matmul.allow_tf32 = True
    model_paths = {"general": args.general_model, "ctc": args.ctc_model}
    models = {
        name: torch.jit.load(str(path), map_location=device).eval()
        for name, path in model_paths.items()
    }
    config = ProbeConfig()
    train_dir = args.competition_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Competition train directory not found: {train_dir}")
    train_paths = _training_paths(train_dir, args.train_per_prefix)

    probes: dict[str, torch.nn.Linear] = {}
    training_evidence: dict[str, dict[str, Any]] = {}
    for name in ("general", "ctc"):
        print(f"HOCT V2 EXTRACTING {name} FEATURES", flush=True)
        features, labels, extraction = extract_training_examples(
            models[name], train_paths, device, config
        )
        if extraction["candidate_recall"] < 0.98:
            raise RuntimeError(f"HOCT {name} candidate recall is too low: {extraction}")
        probe, fitting = fit_probe(features, labels, models[name], device, config)
        probes[name] = probe
        training_evidence[name] = {
            "source_model_sha256": sha256_file(model_paths[name]),
            "extraction": extraction,
            "fitting": fitting,
        }
        del features, labels
        gc.collect()
        torch.cuda.empty_cache()

    checkpoint = {
        "schema_version": 1,
        "model_family": "HOCT multi-backbone linear probes",
        "probe_config": asdict(config),
        "backbones": {
            name: {
                **training_evidence[name],
                "head_weight": probes[name].weight.detach().cpu(),
                "head_bias": probes[name].bias.detach().cpu(),
            }
            for name in sorted(models)
        },
    }
    checkpoint_path = args.output_dir / "hoct_multibackbone_probes.pt"
    torch.save(checkpoint, checkpoint_path)

    processed_videos = read_submission(args.processed_validation_csv)
    raw_videos = read_raw_graphs(args.validation_predictions, set(processed_videos))
    transfer = transfer_raw_edge_probabilities(processed_videos, raw_videos)
    validation = validate_multibackbone_movies(
        models,
        probes,
        processed_videos,
        train_dir,
        device,
        config,
        args.output_dir,
        np.full(3, args.core_size, dtype=np.float32),
    )
    elapsed = time.monotonic() - started
    if elapsed > args.max_wall_seconds:
        raise TimeoutError(f"HOCT V2 exceeded declared wall budget: {elapsed:.1f}s")
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "model_sha256": {
            name: sha256_file(path) for name, path in model_paths.items()
        },
        "probe_checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_config": asdict(config),
        "training_evidence": training_evidence,
        "edge_probability_transfer": transfer,
        "complete_movie_validation_sha256": sha256_file(
            args.output_dir / "complete_movie_validation.json"
        ),
        "selected": validation["selected"],
        "selection_delta_vs_base": validation["selection_delta_vs_base"],
        "acceptance_delta_vs_base": validation["acceptance_delta_vs_base"],
        "association_acceptance_passed": validation["acceptance_passed"],
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
