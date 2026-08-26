"""SwinUNETR heatmap detector initialized from the audited 3D LSM foundation model."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn


EXPECTED_PRETRAINED_PARAMETERS = 16_656_946
EXPECTED_PRETRAINED_TENSORS = 159
EXPECTED_DETECTOR_PARAMETERS = 15_702_979
EXPECTED_SOURCE_SHA256 = "ef0f3d100f9a9aaa5d9a48bb9b07e7f1b0e0d1cc690cdcd308b1e0446bd01634"
OUTPUT_KEYS = {"out.conv.conv.weight", "out.conv.conv.bias"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path, *, expected_sha256: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("stripped LSM-FM checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported stripped LSM-FM checkpoint schema")
    if payload.get("parameter_count") != EXPECTED_PRETRAINED_PARAMETERS:
        raise ValueError("unexpected stripped LSM-FM parameter count")
    source = payload.get("source", {})
    if source.get("checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("stripped LSM-FM source provenance mismatch")
    state = payload.get("state_dict", {})
    if len(state) != EXPECTED_PRETRAINED_TENSORS:
        raise ValueError("unexpected stripped LSM-FM tensor count")
    if sum(value.numel() for value in state.values()) != EXPECTED_PRETRAINED_PARAMETERS:
        raise ValueError("stripped LSM-FM state parameter count mismatch")
    return payload


def build_lsm_fm_detector(checkpoint: Path, *, expected_sha256: str) -> nn.Module:
    """Load all transferable 3D weights and replace only the 512-channel pretext head."""

    from monai.networks.nets import SwinUNETR

    payload = _load_payload(checkpoint, expected_sha256=expected_sha256)
    pretrained = payload["state_dict"]
    if set(pretrained) & OUTPUT_KEYS != OUTPUT_KEYS:
        raise ValueError("LSM-FM checkpoint is missing the expected pretext output head")

    model = SwinUNETR(
        in_channels=1,
        out_channels=1,
        feature_size=24,
        use_checkpoint=True,
        spatial_dims=3,
    )
    transferable = {key: value for key, value in pretrained.items() if key not in OUTPUT_KEYS}
    incompatibility = model.load_state_dict(transferable, strict=False)
    if set(incompatibility.missing_keys) != OUTPUT_KEYS or incompatibility.unexpected_keys:
        raise RuntimeError(
            "LSM-FM transfer mismatch: "
            f"missing={incompatibility.missing_keys}, unexpected={incompatibility.unexpected_keys}"
        )
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected LSM-FM detector parameter count")

    nn.init.normal_(model.out.conv.conv.weight, mean=0.0, std=1e-3)
    nn.init.constant_(model.out.conv.conv.bias, -4.0)
    return model


def set_detector_training_phase(
    model: nn.Module,
    *,
    unfreeze_last_encoder_blocks: int = 0,
) -> dict[str, int]:
    """Train the convolutional decoder first, then the last Swin stages."""

    if not 0 <= unfreeze_last_encoder_blocks <= 4:
        raise ValueError("unfreeze_last_encoder_blocks must lie in [0, 4]")
    model.requires_grad_(True)
    model.swinViT.requires_grad_(False)
    if unfreeze_last_encoder_blocks:
        stage_names = ("layers1", "layers2", "layers3", "layers4")
        for name in stage_names[-unfreeze_last_encoder_blocks:]:
            getattr(model.swinViT, name).requires_grad_(True)

    encoder_trainable = sum(
        parameter.numel()
        for parameter in model.swinViT.parameters()
        if parameter.requires_grad
    )
    decoder_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("swinViT.") and parameter.requires_grad
    )
    return {
        "encoder_trainable_parameters": encoder_trainable,
        "decoder_trainable_parameters": decoder_trainable,
        "total_trainable_parameters": encoder_trainable + decoder_trainable,
    }


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
