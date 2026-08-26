from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from research.hoct_graph.rerank_hoct_multibackbone_submission import (
    load_accepted_multibackbone,
)


class _TinyScriptable(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(288, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(value)


def _config() -> dict:
    return {
        "window_size": 5,
        "window_stride": 4,
        "neighbors": 8,
        "max_distance": 80.0,
        "negative_ratio": 3.0,
        "hard_negative_fraction": 0.6,
        "epochs": 12,
        "batch_size": 8192,
        "learning_rate": 0.002,
        "anchor_weight": 0.0001,
        "weight_decay": 0.00001,
        "max_examples": 600000,
        "seed": 20260826,
    }


def _accepted_fixture(
    tmp_path: Path, variant: str
) -> tuple[dict[str, Path], Path, Path]:
    model_paths = {
        "general": tmp_path / "general_v1.pt",
        "ctc": tmp_path / "ctc_v0.pt",
    }
    model_hashes = {}
    for path in model_paths.values():
        torch.jit.script(_TinyScriptable()).save(str(path))
        model_hashes[
            "general" if path.name.startswith("general") else "ctc"
        ] = hashlib.sha256(path.read_bytes()).hexdigest()
    probe_path = tmp_path / "hoct_multibackbone_probes.pt"
    torch.save(
        {
            "probe_config": _config(),
            "backbones": {
                name: {
                    "source_model_sha256": model_hashes[name],
                    "head_weight": torch.zeros((1, 288)),
                    "head_bias": torch.zeros(1),
                }
                for name in model_paths
            },
        },
        probe_path,
    )
    terminal_path = tmp_path / "training_terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "association_acceptance_passed": True,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "model_sha256": model_hashes,
                "probe_checkpoint_sha256": hashlib.sha256(
                    probe_path.read_bytes()
                ).hexdigest(),
                "selected": {
                    "variant": variant,
                    "method": "raw_confidence_hybrid",
                },
            }
        ),
        encoding="utf-8",
    )
    return model_paths, probe_path, terminal_path


def test_single_backbone_winner_loads_only_required_model(tmp_path: Path) -> None:
    paths = _accepted_fixture(tmp_path, "ctc:biohub_probe")

    models, probes, config, terminal = load_accepted_multibackbone(
        *paths, device=torch.device("cpu")
    )

    assert set(models) == {"ctc"}
    assert set(probes) == {"ctc"}
    assert config.neighbors == 8
    assert terminal["selected"]["variant"] == "ctc:biohub_probe"


def test_blend_winner_loads_both_models(tmp_path: Path) -> None:
    paths = _accepted_fixture(
        tmp_path, "general_ctc_blend_w0.5:pretrained"
    )

    models, probes, _config_value, _terminal = load_accepted_multibackbone(
        *paths, device=torch.device("cpu")
    )

    assert set(models) == {"general", "ctc"}
    assert set(probes) == {"general", "ctc"}


def test_multibackbone_probe_mutation_is_rejected(tmp_path: Path) -> None:
    model_paths, probe_path, terminal_path = _accepted_fixture(
        tmp_path, "general:pretrained"
    )
    probe_path.write_bytes(probe_path.read_bytes() + b"mutation")

    with pytest.raises(RuntimeError, match="different probe"):
        load_accepted_multibackbone(
            model_paths,
            probe_path,
            terminal_path,
            device=torch.device("cpu"),
        )
