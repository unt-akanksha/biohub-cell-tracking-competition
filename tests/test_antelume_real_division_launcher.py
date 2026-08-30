from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run-antelume-real-division-gate-head-v1.sh"
FOCUSED_SCRIPT = ROOT / "scripts/run-antelume-real-division-gate-focused-v1.sh"


def test_launcher_is_hash_bound_antelume_only_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "A10G" in source
    assert "kaggle" not in source.lower()
    assert "46db151310808fa455e63a4c8791f14121125ba6bda0d46b5aa2b1d7913990e0" in source
    assert "943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e" in source
    assert "--train-mode head_only" in source
    assert "--training-terminal" in source
    assert "authorized_for_competition_graph_evaluation" in source
    assert "authorized_for_submission\") is False" in source


def test_launcher_never_touches_unrelated_antelume_workspace() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "/home/ubuntu/antelume" not in source
    assert "/home/ubuntu/biohub" in source
    assert "rm -rf" not in source


def test_focused_launcher_is_bound_to_rejected_head_checkpoints() -> None:
    source = FOCUSED_SCRIPT.read_text(encoding="utf-8")

    assert "--train-mode focused" in source
    assert "--steps 3000" in source
    assert "ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133" in source
    assert "cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75" in source
    assert "A10G" in source
    assert "kaggle" not in source.lower()
    assert "rm -rf" not in source
