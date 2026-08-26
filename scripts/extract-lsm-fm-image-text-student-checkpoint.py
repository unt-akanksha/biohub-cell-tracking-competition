#!/usr/bin/env python
"""Safely strip the licensed feature-36 LSM-FM student from its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "f08122ee471fae0c555f3b71df37087d6dd78ffda7bf000f0510aa262c1c5d15"
EXPECTED_STUDENT_PARAMETERS = 36_032_614
EXPECTED_STUDENT_TENSORS = 159


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allow_checkpoint_metadata(torch) -> None:
    """Allow only the explicit metadata globals present in the audited checkpoint."""

    import _codecs

    import numpy as np
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

    dtype_classes = {
        type(np.dtype(name))
        for name in (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
        )
    }
    torch.serialization.add_safe_globals(
        [
            MetaTensor,
            (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
            _codecs.encode,
            MetaKeys,
            SpaceKeys,
            TraceKeys,
            *dtype_classes,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("multimodal LSM-FM source checkpoint hash mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import torch

    allow_checkpoint_metadata(torch)
    checkpoint = torch.load(
        args.source,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state = {
        key.removeprefix("student_encoder."): value.detach().cpu()
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("student_encoder.")
    }
    parameter_count = sum(value.numel() for value in state.values())
    if len(state) != EXPECTED_STUDENT_TENSORS:
        raise RuntimeError("unexpected multimodal LSM-FM student tensor count")
    if parameter_count != EXPECTED_STUDENT_PARAMETERS:
        raise RuntimeError("unexpected multimodal LSM-FM student parameter count")

    payload = {
        "schema_version": 1,
        "state_dict": state,
        "architecture": {
            "name": "MONAI SwinUNETR",
            "in_channels": 1,
            "out_channels": 512,
            "feature_size": 36,
            "spatial_dims": 3,
            "pretraining_input_shape": [64, 64, 64],
            "pretraining_objective": "image-text plus masked reconstruction and distillation",
        },
        "parameter_count": parameter_count,
        "source": {
            "checkpoint_sha256": SOURCE_SHA256,
            "doi": "10.5281/zenodo.20146516",
            "license": "CC-BY-4.0",
            "repository_commit": "cb772551652c3ee762048936e396743fe9a65e10",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "parameter_count": parameter_count,
            "tensor_count": len(state),
        }
    )


if __name__ == "__main__":
    main()
