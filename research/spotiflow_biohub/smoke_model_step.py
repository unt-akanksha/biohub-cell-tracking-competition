#!/usr/bin/env python
"""Run one optimizer step through an official Spotiflow 3D checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    # The release wheel carries Linux-only peak-extraction extensions. They are
    # irrelevant to a forward/backward smoke on Windows, so provide fail-fast
    # import stubs instead of pretending to test peak inference locally.
    extension_exports = {
        "filters": ("c_maximum_filter_2d_float",),
        "filters3d": ("c_maximum_filter_3d_float",),
        "point_nms": ("c_point_nms_2d",),
        "point_nms3d": ("c_point_nms_3d",),
        "spotflow2d": ("c_gaussian2d", "c_gaussian2d_sum", "c_spotflow2d"),
        "spotflow3d": ("c_gaussian3d", "c_spotflow3d"),
    }
    library = types.ModuleType("spotiflow.lib")
    library.__path__ = []
    sys.modules.setdefault("spotiflow.lib", library)
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = wandb_stub

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("compiled peak extension is unavailable in Windows smoke")

    for name, exports in extension_exports.items():
        module = types.ModuleType(f"spotiflow.lib.{name}")
        for export in exports:
            setattr(module, export, unavailable)
        sys.modules.setdefault(module.__name__, module)

    import torch
    from spotiflow.model import Spotiflow

    torch.manual_seed(20260826)
    started = time.monotonic()
    model = Spotiflow.from_folder(
        str(args.model), inference_mode=False, map_location="cpu"
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    image = torch.rand((1, 1, 32, 64, 64), dtype=torch.float32)
    output = model(image)
    loss = sum(value.square().mean() for value in output["heatmaps"])
    if "flow" in output:
        loss = loss + 0.01 * output["flow"].square().mean()
    loss.backward()
    gradient_tensors = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()
    result = {
        "status": "passed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model_sha256": sha256_file(args.model / "best.pt"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(image.shape),
        "heatmap_shapes": [list(value.shape) for value in output["heatmaps"]],
        "flow_shape": list(output["flow"].shape) if "flow" in output else None,
        "loss": float(loss.detach()),
        "parameter_tensors_with_finite_gradients": int(gradient_tensors),
    }
    if result["parameter_count"] != 35_489_892 or gradient_tensors <= 0:
        raise RuntimeError(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
