from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "research/temporal_contrastive/train_graph_context_fresh_ensemble_v3.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("fresh_ensemble_v3", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_and_training_arguments() -> None:
    module = load_module()
    split, roles = module.load_contract(
        ROOT / "research/graph_context_fresh_split_v3.json"
    )
    arguments = module.frozen_training_args()
    assert split["audit_model_predictions_generated"] is False
    assert len(roles) == 146
    assert set(roles.values()) == {"optimization", "selection", "audit"}
    assert arguments.steps == 20_000
    assert arguments.gradient_clip == 2.0
    assert len(module.member_specs()) == 4


def test_audit_is_opened_only_after_policy_is_persisted() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    policy_write = source.index("atomic_json(policy_path, selection_policy)")
    audit_comment = source.index(
        "# This is the first point at which any new-audit shard is opened."
    )
    audit_load = source.index(
        'load_fresh_role(args.data_root, source, roles, "audit")'
    )
    assert policy_write < audit_comment < audit_load


def test_consensus_is_top_rank_agreement_without_absolute_threshold() -> None:
    module = load_module()
    inventory = [
        {"stem": "44b6_a", "embryo": "44b6", "timepoint": 1, "source_row": 0},
        {"stem": "44b6_a", "embryo": "44b6", "timepoint": 1, "source_row": 1},
        {"stem": "6bba_b", "embryo": "6bba", "timepoint": 2, "source_row": 0},
        {"stem": "6bba_b", "embryo": "6bba", "timepoint": 2, "source_row": 1},
    ]
    result = module.consensus_audit(
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([-20.0, -21.0, 100.0, 99.0]),
        torch.tensor([-50.0, -51.0, 0.001, 0.002]),
        inventory,
    )
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["selected"] == 1


def test_generated_kernel_has_no_submission_or_competition_source() -> None:
    metadata = json.loads(
        (
            ROOT
            / "kaggle/biohub-graph-context-fresh-training-v3/kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (
            ROOT
            / "kaggle/biohub-graph-context-fresh-training-v3"
            / "biohub-graph-context-fresh-training-v3.ipynb"
        ).read_text(encoding="ascii")
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == []
    assert len(metadata["dataset_sources"]) == 1
    assert "kaggle competitions submit" not in source.lower()
    assert "submission.csv" not in source.lower()
