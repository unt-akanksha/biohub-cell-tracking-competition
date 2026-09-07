from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

from research.peak_rank_detection import evaluate_peak_rank_detector as evaluation
from research.peak_rank_detection.model import TemporalPeakRankDetector, count_parameters


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_flags() -> dict:
    return {
        "status": "accepted_at_audit",
        "selection_passed": True,
        "audit_opened": True,
        "audit_passed": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def test_equal_logit_ensemble_strict_loads_independent_members(tmp_path: Path) -> None:
    contracts = []
    for index in range(2):
        torch.manual_seed(100 + index)
        model = TemporalPeakRankDetector(
            widths=(4, 8, 16, 32), depths=(1, 1, 1, 1)
        )
        path = tmp_path / f"member-{index}.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        contracts.append(
            {
                "name": f"member-{index}",
                "checkpoint_file": path.name,
                "checkpoint_sha256": sha256(path),
                "parameter_count": count_parameters(model),
                "widths": [4, 8, 16, 32],
                "depths": [1, 1, 1, 1],
            }
        )
    checkpoint = tmp_path / "peak_rank_detector.pt"
    checkpoint.write_text(json.dumps({"members": contracts}), encoding="utf-8")
    terminal = {
        **accepted_flags(),
        "parameter_count": sum(row["parameter_count"] for row in contracts),
        "widths": [row["widths"] for row in contracts],
        "depths": [row["depths"] for row in contracts],
        "ensemble_members": contracts,
        "checkpoint_sha256": sha256(checkpoint),
    }
    terminal_path = tmp_path / "training_terminal.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    verified = evaluation.validate_training(checkpoint, terminal_path)
    ensemble = evaluation.load_model(checkpoint, verified, torch.device("cpu"))
    output = ensemble(torch.rand((1, 3, 8, 8, 8)))
    assert count_parameters(ensemble) == terminal["parameter_count"]
    assert output["logits"].shape == (1, 1, 8, 8, 8)
    assert output["offsets"].shape == (1, 3, 8, 8, 8)


class FixedMember(nn.Module):
    def __init__(self, logits: torch.Tensor, offsets: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)
        self.register_buffer("fixed_offsets", offsets)

    def forward(self, frames: torch.Tensor) -> dict:
        del frames
        return {
            "logits": self.fixed_logits,
            "offsets": self.fixed_offsets,
            "auxiliary_logits": (self.fixed_logits[..., ::2, ::2, ::2],),
        }


def test_confidence_max_ensemble_uses_winning_offsets_per_voxel() -> None:
    left_logits = torch.tensor([[[[[3.0, -1.0]]]]])
    right_logits = torch.tensor([[[[[1.0, 4.0]]]]])
    left_offsets = torch.full((1, 3, 1, 1, 2), 0.25)
    right_offsets = torch.full((1, 3, 1, 1, 2), -0.25)
    ensemble = evaluation.PeakRankConfidenceMaxEnsemble(
        (FixedMember(left_logits, left_offsets), FixedMember(right_logits, right_offsets))
    )

    output = ensemble(torch.empty(0))

    assert torch.equal(output["logits"], torch.tensor([[[[[3.0, 4.0]]]]]))
    assert torch.equal(output["offsets"][..., 0], left_offsets[..., 0])
    assert torch.equal(output["offsets"][..., 1], right_offsets[..., 1])


def write_member_fixture(root: Path, name: str, checkpoint: bytes) -> tuple[dict, Path]:
    runtime = root / f"runtime-{name}"
    runtime.mkdir(parents=True)
    checkpoint_path = runtime / "peak_rank_detector.pt"
    checkpoint_path.write_bytes(checkpoint)
    terminal = {
        **accepted_flags(),
        "parameter_count": 38_381_478,
        "widths": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    terminal_path = runtime / "training_terminal.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (checkpoint_path, terminal_path)
    }
    manifest = {
        "schema_version": 1,
        "architecture": "independent temporal 3D ConvNeXt U-Net peak ranker",
        "parameter_count": 38_381_478,
        "widths": terminal["widths"],
        "depths": terminal["depths"],
        "ensemble_size": 1,
        "training_audit_passed": True,
        "checkpoint_sha256": sha256(checkpoint_path),
        "files": files,
    }
    manifest_path = runtime / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    download = root / f"download-{name}"
    download.mkdir()
    result = {
        "run_id": "temporal-peak-rank-clean-validation-v1",
        "selection_passed": True,
        "selected_tta_mode": "none",
        "acceptance_opened": True,
        "promotion_passed": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "provenance": {"checkpoint_sha256": sha256(checkpoint_path)},
    }
    result_path = download / "peak_rank_validation.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    controller_path = root / f"controller-{name}.json"
    controller = {
        "status": "completed",
        "accepted_for_candidate_integration": True,
        "promotion_passed": True,
        "competition_submission_performed": False,
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation_result_sha256": sha256(result_path),
        "download_root": str(download),
    }
    controller_path.write_text(json.dumps(controller), encoding="utf-8")
    return {
        "name": name,
        "runtime": runtime,
        "controller": controller_path,
        "checkpoint_file": f"member-{name}.pt",
    }, manifest_path


def test_runtime_builder_requires_two_clean_distinct_members(
    tmp_path: Path, monkeypatch
) -> None:
    spec = importlib.util.spec_from_file_location("ensemble_runtime_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    first, _ = write_member_fixture(tmp_path, "v1", b"member one")
    second, _ = write_member_fixture(tmp_path, "v2", b"member two")
    source = tmp_path / "model.py"
    source.write_text("# packaged source\n", encoding="utf-8")
    target_name = "biohub-peak-rank-logit-ensemble-validation-runtime-v4"
    target = tmp_path / ".biohub" / "staging" / target_name
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "TARGET", target)
    monkeypatch.setattr(module, "TARGET_NAME", target_name)
    monkeypatch.setattr(module, "MEMBERS", (first, second))
    monkeypatch.setattr(module, "SOURCES", {"model.py": source})
    monkeypatch.setattr(sys, "argv", [str(BUILDER)])
    module.main()

    terminal = json.loads((target / "training_terminal.json").read_text())
    manifest = json.loads((target / "SOURCE_MANIFEST.json").read_text())
    assert terminal["parameter_count"] == 76_762_956
    assert len(terminal["ensemble_members"]) == 2
    assert manifest["ensemble_size"] == 2
    assert manifest["checkpoint_sha256"] == sha256(target / "peak_rank_detector.pt")
    assert (target / "member-v1.pt").read_bytes() == b"member one"
    assert (target / "member-v2.pt").read_bytes() == b"member two"
