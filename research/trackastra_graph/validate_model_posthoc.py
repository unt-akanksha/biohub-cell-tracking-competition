from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from trainer import complete_movie_validation
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.trackastra_graph.train_biohub_graph_transformer import (
        complete_movie_validation,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_source_terminal(model_dir: Path) -> tuple[dict[str, Any], str]:
    terminal_path = model_dir / "training_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed":
        raise RuntimeError("Source association training did not complete")
    model_path = model_dir / "model.pt"
    model_sha256 = sha256_file(model_path)
    if model_sha256 != terminal.get("model_sha256"):
        raise RuntimeError("Source checkpoint does not match its training terminal")
    if not (model_dir / "config.yaml").is_file():
        raise FileNotFoundError("Source Trackastra config.yaml is missing")
    return terminal, model_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trackastra-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--candidate-radius", type=float, default=80.0)
    args = parser.parse_args()

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_terminal, model_sha256 = validate_source_terminal(args.model_dir)

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for complete-movie posthoc acceptance")
    model = TrackingTransformer.from_folder(args.model_dir, map_location="cpu").to(
        device
    )
    model.eval()
    train_dir = args.competition_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Competition train directory not found: {train_dir}")

    complete = complete_movie_validation(
        model,
        args.validation_predictions,
        train_dir,
        device,
        args.output_dir,
        args.max_tokens,
        args.candidate_radius,
    )
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_kind": "posthoc_complete_movie_acceptance",
        "elapsed_seconds": time.monotonic() - started,
        "model_sha256": model_sha256,
        "source_training_terminal_sha256": sha256_file(
            args.model_dir / "training_terminal.json"
        ),
        "source_training_acceptance_passed": bool(
            source_terminal.get("association_acceptance_passed", False)
        ),
        "complete_movie_validation_sha256": sha256_file(
            args.output_dir / "complete_movie_validation.json"
        ),
        "complete_movie_selected": complete["selected"],
        "complete_movie_base_raw": complete["base_raw_graph"],
        "stored_edge_probability_evidence": complete[
            "stored_edge_probability_evidence"
        ],
        "selection_delta_vs_base_raw": complete["selection_delta_vs_base_raw"],
        "acceptance_delta_vs_base_raw": complete["acceptance_delta_vs_base_raw"],
        "association_acceptance_passed": complete["acceptance_passed"],
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.output_dir / "acceptance_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
