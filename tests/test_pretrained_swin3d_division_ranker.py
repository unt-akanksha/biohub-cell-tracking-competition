from pathlib import Path
import runpy

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "research/temporal_contrastive/train_pretrained_swin3d_division_ranker.py"
)
LAUNCHER = ROOT / "scripts/run-antelume-pretrained-swin3d-division-ranker-v1.sh"
LAUNCHER_V2 = ROOT / "scripts/run-antelume-pretrained-swin3d-division-ranker-v2.sh"


def test_rank_loss_prefers_positive_scores_above_negative_scores() -> None:
    module = runpy.run_path(str(TRAINER))
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = module["rank_loss"](
        torch.tensor([2.0, 1.0, -1.0, -2.0]), targets
    )
    bad = module["rank_loss"](
        torch.tensor([-2.0, -1.0, 1.0, 2.0]), targets
    )

    assert good < bad


def test_nested_tuning_and_audit_movies_are_disjoint() -> None:
    module = runpy.run_path(str(TRAINER))
    inventory = []
    targets = []
    for embryo in ("44b6", "6bba"):
        for movie in range(10):
            stem = f"{embryo}_{movie:02d}"
            inventory.extend(
                [
                    {"stem": stem, "embryo": embryo},
                    {"stem": stem, "embryo": embryo},
                ]
            )
            targets.extend((1.0, 0.0))

    tuning, audit = module["nested_tuning_and_audit_stems"](
        inventory, torch.tensor(targets), seed=7
    )

    assert tuning
    assert audit
    assert not tuning & audit


def test_ranker_contract_is_large_threshold_free_and_non_submitting() -> None:
    module = runpy.run_path(str(TRAINER))
    source = TRAINER.read_text(encoding="utf-8")

    assert module["EXPECTED_PARAMETER_COUNT"] == 87_640_009
    assert module["RUN_ID"] == "competition-pretrained-swin3d-division-ranker-v2"
    assert module["EXPECTED_TRAINABLE_PARAMETERS"] == 25_357_761
    assert module["TRAINABLE_PREFIXES"] == ("features.6.", "norm.", "head.")
    assert module["PRETRAINED_WEIGHTS"] == "Swin3D_B_Weights.KINETICS400_V1"
    assert '"absolute_threshold_used": False' in source
    assert '"authorized_for_submission": False' in source
    assert "competition_test_data_read\": True" not in source


def test_antelume_launcher_is_gpu_bound_and_non_submitting() -> None:
    source = LAUNCHER_V2.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "--required-gpu-name A10G" in source
    assert "--steps 1500" in source
    assert "competition-pretrained-swin3d-division-ranker-v2" in source
    assert "kaggle competitions submit" not in source
