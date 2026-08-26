from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MONAI_DEPS = ROOT / ".biohub" / "cache" / "spatialdino-deps"
CHECKPOINT = (
    ROOT
    / ".biohub"
    / "cache"
    / "models"
    / "lsm-fm"
    / "lsm_fm_image_text_student.pt"
)
if str(MONAI_DEPS) not in sys.path:
    sys.path.insert(0, str(MONAI_DEPS))

from research.lsm_fm_detection.image_text_model import (  # noqa: E402
    ARCHITECTURE_DESCRIPTION,
    EXPECTED_DETECTOR_PARAMETERS,
    EXPECTED_PRETRAINED_PARAMETERS,
    build_lsm_fm_detector,
    set_detector_training_phase,
)


CHECKPOINT_SHA256 = "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7"


def test_image_text_checkpoint_transfers_every_non_head_tensor() -> None:
    assert "image-text" in ARCHITECTURE_DESCRIPTION
    assert "feature-36" in ARCHITECTURE_DESCRIPTION
    model = build_lsm_fm_detector(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_DETECTOR_PARAMETERS
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    assert sum(value.numel() for value in source.values()) == EXPECTED_PRETRAINED_PARAMETERS
    torch.testing.assert_close(
        model.swinViT.patch_embed.proj.weight,
        source["swinViT.patch_embed.proj.weight"],
    )
    torch.testing.assert_close(
        model.decoder1.conv_block.conv3.conv.weight,
        source["decoder1.conv_block.conv3.conv.weight"],
    )
    torch.testing.assert_close(
        model.out.conv.conv.bias,
        torch.full_like(model.out.conv.conv.bias, -4.0),
    )


def test_image_text_training_phase_has_expected_capacity() -> None:
    model = build_lsm_fm_detector(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    assert set_detector_training_phase(model, unfreeze_last_encoder_blocks=0) == {
        "encoder_trainable_parameters": 0,
        "decoder_trainable_parameters": 30_445_381,
        "total_trainable_parameters": 30_445_381,
    }
    assert set_detector_training_phase(model, unfreeze_last_encoder_blocks=4) == {
        "encoder_trainable_parameters": 4_626_810,
        "decoder_trainable_parameters": 30_445_381,
        "total_trainable_parameters": 35_072_191,
    }
