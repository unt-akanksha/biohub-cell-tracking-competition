from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MONAI_DEPS = ROOT / ".biohub" / "cache" / "spatialdino-deps"
CHECKPOINT = (
    ROOT / ".biohub" / "cache" / "models" / "lsm-fm" / "lsm_fm_image_only_student.pt"
)
if str(MONAI_DEPS) not in sys.path:
    sys.path.insert(0, str(MONAI_DEPS))

from research.lsm_fm_detection.model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS,
    build_lsm_fm_detector,
    set_detector_training_phase,
)


CHECKPOINT_SHA256 = "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"


def test_lsm_fm_checkpoint_transfers_every_non_head_tensor() -> None:
    model = build_lsm_fm_detector(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_DETECTOR_PARAMETERS
    torch.testing.assert_close(
        model.out.conv.conv.bias,
        torch.full_like(model.out.conv.conv.bias, -4.0),
    )
    assert float(model.out.conv.conv.weight.detach().std()) < 0.002

    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    torch.testing.assert_close(
        model.swinViT.patch_embed.proj.weight,
        source["swinViT.patch_embed.proj.weight"],
    )
    torch.testing.assert_close(
        model.decoder1.conv_block.conv3.conv.weight,
        source["decoder1.conv_block.conv3.conv.weight"],
    )


def test_lsm_fm_training_phase_freezes_then_unfreezes_only_tail_stages() -> None:
    model = build_lsm_fm_detector(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    frozen = set_detector_training_phase(model, unfreeze_last_encoder_blocks=0)
    assert frozen["encoder_trainable_parameters"] == 0
    assert frozen["decoder_trainable_parameters"] > 0
    assert all(not parameter.requires_grad for parameter in model.swinViT.parameters())

    tail = set_detector_training_phase(model, unfreeze_last_encoder_blocks=2)
    assert tail["encoder_trainable_parameters"] > 0
    assert all(
        parameter.requires_grad
        for name in ("layers3", "layers4")
        for parameter in getattr(model.swinViT, name).parameters()
    )
    assert all(
        not parameter.requires_grad
        for name in ("patch_embed", "layers1", "layers2")
        for parameter in getattr(model.swinViT, name).parameters()
    )
