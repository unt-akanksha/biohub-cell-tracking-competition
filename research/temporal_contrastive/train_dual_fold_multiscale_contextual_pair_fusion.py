#!/usr/bin/env python
"""Transfer the distinct multiscale contextual v4 on exactly two GPUs.

The wrapper preserves the v3 transition adapter, bidirectional candidate loss,
reciprocal embryo folds, validation/calibration split, synthetic-retention gate,
and no-submission boundary. It accepts only hash-bound v4 ZebraHub pretraining
that itself records an accepted v3 warm start.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

try:
    import train_dual_fold_contextual_pair_fusion as contextual
    import train_dual_fold_pair_fusion as generic
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from train_zebrahub_multiscale_contextual_pretrain import (
        RUN_ID as PRETRAINING_RUN_ID,
        multiscale_family_metadata,
    )
    from train_zebrahub_contextual_pretrain import VALIDATION_PARTITION_POLICY
except ModuleNotFoundError:
    from research.temporal_contrastive import (
        train_dual_fold_contextual_pair_fusion as contextual,
    )
    from research.temporal_contrastive import train_dual_fold_pair_fusion as generic
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_zebrahub_multiscale_contextual_pretrain import (
        RUN_ID as PRETRAINING_RUN_ID,
        multiscale_family_metadata,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        VALIDATION_PARTITION_POLICY,
    )


RUN_ID = "temporal-multiscale-contextual-pair-fusion-v4"


def multiscale_contextual_family_metadata() -> dict[str, object]:
    return {
        **contextual.contextual_family_metadata(),
        **multiscale_family_metadata(),
    }


def load_multiscale_initial_model(
    model: torch.nn.Module,
    root: Path | None,
    fold: str,
) -> dict[str, object]:
    if not isinstance(model, MultiscaleContextualPairFusionAssociationModel):
        raise TypeError("v4 transfer model factory changed")
    if root is None:
        raise ValueError("multiscale v4 requires hash-bound ZebraHub pretraining")
    root = root.resolve()
    aggregate_path = root / "pretraining_terminal.json"
    worker_path = root / fold / "worker_terminal.json"
    model_path = root / fold / "pretrained_model.pt"
    if not all(path.is_file() for path in (aggregate_path, worker_path, model_path)):
        raise FileNotFoundError(f"incomplete multiscale pretraining evidence for {fold}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    model_hash = generic.base.sha256_file(model_path)
    initialization = worker.get("initialization", {})
    valid = bool(
        aggregate.get("status") == "completed"
        and aggregate.get("run_id") == PRETRAINING_RUN_ID
        and aggregate.get("appearance_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_improved") is True
        and aggregate.get("submission_created") is False
        and worker.get("status") == "completed"
        and worker.get("run_id") == aggregate.get("run_id")
        and worker.get("fold") == fold
        and worker.get("appearance_family") == aggregate.get("appearance_family")
        and int(worker.get("parameter_count", 0)) == EXPECTED_PARAMETER_COUNT
        and int(worker.get("best_step", 0)) > 0
        and worker.get("selection_gate_passed") is True
        and worker.get("audit_gate_passed") is True
        and worker.get("validation_partition_policy")
        == VALIDATION_PARTITION_POLICY
    )
    # Keep security/provenance checks outside the expression above so the
    # exact policy is readable and cannot be weakened by truthiness precedence.
    valid = bool(
        valid
        and worker.get("model_sha256") == model_hash
        and worker.get("external_training_source") == "ZSNS004"
        and worker.get("external_validation_source") == "ZSNS005"
        and worker.get("competition_data_read") is False
        and worker.get("public_predictions_copied") is False
        and worker.get("public_leaderboard_used_for_selection") is False
        and worker.get("submission_created") is False
        and initialization.get("run_id") == "zebrahub-contextual-pretrain-v1"
        and initialization.get("source_family")
        == "temporal_contextual_pair_fusion_v3"
        and initialization.get("target_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and initialization.get("initial_predictions_numerically_preserved") is True
    )
    if not valid:
        raise ValueError(f"invalid multiscale pretraining evidence for {fold}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return {
        "policy": "hash-bound multiscale ZebraHub external pretraining",
        "run_id": aggregate["run_id"],
        "appearance_family": aggregate["appearance_family"],
        "fold": fold,
        "model_sha256": model_hash,
        "external_training_source": "ZSNS004",
        "external_validation_source": "ZSNS005",
        "v3_warm_start_run_id": initialization["run_id"],
    }


def configure() -> None:
    contextual.configure()
    generic.RUN_ID = RUN_ID
    generic.APPEARANCE_FAMILY = MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    generic.MODEL_CLASS = MultiscaleContextualPairFusionAssociationModel
    generic.EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    generic.WORKER_SCRIPT_PATH = Path(__file__).resolve()
    generic.family_metadata = multiscale_contextual_family_metadata
    generic.load_initial_model = load_multiscale_initial_model


def main() -> None:
    contextual.require_external_pretraining(sys.argv[1:])
    configure()
    generic.main()


if __name__ == "__main__":
    main()
