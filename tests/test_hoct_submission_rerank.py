from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from research.hoct_graph.rerank_hoct_submission import load_accepted_probe


class _TinyScriptable(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(288, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(value)


def _accepted_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "general_v1.pt"
    torch.jit.script(_TinyScriptable()).save(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    probe_path = tmp_path / "hoct_probe.pt"
    torch.save(
        {
            "source_model_sha256": model_sha,
            "head_weight": torch.zeros((1, 288)),
            "head_bias": torch.zeros(1),
            "probe_config": {
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
                "source_model_sha256": model_sha,
                "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
                "selected": {
                    "variant": "biohub_probe",
                    "method": "raw_confidence_hybrid",
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path, probe_path, terminal_path


def test_accepted_probe_is_hash_bound_to_backbone_and_head(tmp_path: Path) -> None:
    paths = _accepted_fixture(tmp_path)
    _model, probe, config, terminal = load_accepted_probe(
        *paths, device=torch.device("cpu")
    )
    assert probe.weight.shape == (1, 288)
    assert config.neighbors == 8
    assert terminal["selected"]["variant"] == "biohub_probe"


def test_probe_mutation_is_rejected(tmp_path: Path) -> None:
    model_path, probe_path, terminal_path = _accepted_fixture(tmp_path)
    probe_path.write_bytes(probe_path.read_bytes() + b"mutation")
    with pytest.raises(RuntimeError, match="different probe"):
        load_accepted_probe(
            model_path,
            probe_path,
            terminal_path,
            device=torch.device("cpu"),
        )
