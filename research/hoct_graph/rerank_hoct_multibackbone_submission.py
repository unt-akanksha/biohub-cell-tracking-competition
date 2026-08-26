from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from multibackbone import materialize_variant, required_backbones
    from rerank_submission import (
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
        validate_edges,
        write_submission,
    )
    from train_biohub_hoct_probe import (
        ProbeConfig,
        link_configuration,
        predict_movie_variants,
        sha256_file,
    )
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.hoct_graph.multibackbone import (
        materialize_variant,
        required_backbones,
    )
    from research.hoct_graph.train_biohub_hoct_probe import (
        ProbeConfig,
        link_configuration,
        predict_movie_variants,
        sha256_file,
    )
    from research.trackastra_graph.rerank_submission import (
        read_raw_graphs,
        read_submission,
        transfer_raw_edge_probabilities,
        validate_edges,
        write_submission,
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_accepted_multibackbone(
    model_paths: dict[str, Path],
    probe_path: Path,
    terminal_path: Path,
    device: torch.device,
) -> tuple[
    dict[str, torch.jit.ScriptModule],
    dict[str, torch.nn.Linear],
    ProbeConfig,
    dict[str, Any],
]:
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed":
        raise RuntimeError("HOCT V2 source training did not complete")
    if not terminal.get("association_acceptance_passed", False):
        raise RuntimeError("HOCT V2 failed its clean held-out acceptance gate")
    if terminal.get("public_leaderboard_used_for_selection") is not False:
        raise RuntimeError("HOCT V2 lacks a no-leaderboard selection assertion")
    if terminal.get("submission_created") is not False:
        raise RuntimeError("HOCT V2 training unexpectedly created a submission")
    if terminal.get("probe_checkpoint_sha256") != sha256_file(probe_path):
        raise RuntimeError("Accepted HOCT V2 terminal is bound to a different probe")

    selected = terminal.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("variant"), str):
        raise RuntimeError("Accepted HOCT V2 terminal lacks a selected variant")
    needed = required_backbones(selected["variant"])
    if set(model_paths) != {"general", "ctc"}:
        raise RuntimeError("Both official HOCT model paths must be supplied")
    actual_model_hashes = {
        name: sha256_file(path) for name, path in model_paths.items()
    }
    if terminal.get("model_sha256") != actual_model_hashes:
        raise RuntimeError("Accepted HOCT V2 terminal is bound to different backbones")

    checkpoint = torch.load(probe_path, map_location="cpu", weights_only=True)
    config = ProbeConfig(**checkpoint["probe_config"])
    stored = checkpoint.get("backbones")
    if not isinstance(stored, dict) or set(stored) != {"general", "ctc"}:
        raise RuntimeError("HOCT V2 probe checkpoint lacks both backbones")
    models: dict[str, torch.jit.ScriptModule] = {}
    probes: dict[str, torch.nn.Linear] = {}
    for name in sorted(needed):
        record = stored[name]
        if record.get("source_model_sha256") != actual_model_hashes[name]:
            raise RuntimeError(f"HOCT V2 {name} probe is bound to another backbone")
        weight = record["head_weight"].detach().float().reshape(1, -1)
        bias = record["head_bias"].detach().float().reshape(1)
        if weight.shape != (1, 288) or bias.shape != (1,):
            raise RuntimeError(f"HOCT V2 {name} probe has invalid dimensions")
        probe = torch.nn.Linear(288, 1)
        probe.weight.data.copy_(weight)
        probe.bias.data.copy_(bias)
        probes[name] = probe.to(device).eval()
        models[name] = torch.jit.load(str(model_paths[name]), map_location=device).eval()
    return models, probes, config, terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--raw-graph-root", type=Path, required=True)
    parser.add_argument("--general-model", type=Path, required=True)
    parser.add_argument("--ctc-model", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--acceptance-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--core-size", type=float, default=128.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for dense HOCT V2 submission inference")
    model_paths = {"general": args.general_model, "ctc": args.ctc_model}
    models, probes, config, terminal = load_accepted_multibackbone(
        model_paths, args.probes, args.acceptance_terminal, device
    )
    videos = read_submission(args.base_submission)
    raw_videos = read_raw_graphs(args.raw_graph_root, set(videos))
    transfer = transfer_raw_edge_probabilities(videos, raw_videos)
    selected = terminal["selected"]
    selected_variant = str(selected["variant"])
    edges_by_stem: dict[str, list[tuple[int, int]]] = {}
    changes: dict[str, dict[str, int]] = {}
    core_size = np.full(3, args.core_size, dtype=np.float32)
    for stem, video in sorted(videos.items()):
        print(f"HOCT V2 SUBMISSION INFERENCE: {stem}", flush=True)
        by_backbone = {
            name: predict_movie_variants(
                model, probes[name], video, device, config, core_size=core_size
            )
            for name, model in models.items()
        }
        scores = materialize_variant(selected_variant, by_backbone)
        edges = link_configuration(video, scores, selected)
        validate_edges(video, edges)
        base = {tuple(map(int, edge)) for edge in video.edges.tolist()}
        candidate = set(edges)
        changes[stem] = {
            "base_edges": len(base),
            "candidate_edges": len(candidate),
            "retained_edges": len(base & candidate),
            "removed_edges": len(base - candidate),
            "added_edges": len(candidate - base),
        }
        edges_by_stem[stem] = edges
    total_changes = sum(
        row["removed_edges"] + row["added_edges"] for row in changes.values()
    )
    if total_changes == 0:
        raise RuntimeError("HOCT V2 candidate is edge-identical to the public comparator")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, videos, edges_by_stem)
    report = {
        "schema_version": 1,
        "status": "completed",
        "candidate_kind": "accepted_HOCT_multibackbone_probability_aware_rerank",
        "base_submission_sha256": sha256_file(args.base_submission),
        "output_submission_sha256": sha256_file(args.output),
        "source_model_sha256": {
            name: sha256_file(path) for name, path in model_paths.items()
        },
        "probe_checkpoint_sha256": sha256_file(args.probes),
        "acceptance_terminal_sha256": sha256_file(args.acceptance_terminal),
        "selected": selected,
        "raw_probability_transfer": transfer,
        "changes": changes,
        "total_edge_changes": total_changes,
        "node_rows_preserved": True,
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
