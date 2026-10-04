"""Twenty-minute, label-free, paired public encoder/DeepCenter GPU check.

Run only on a GPU the user has freed. No pip installs, training, Kaggle calls,
graph generation, label loading, process killing, or competition submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args, result):
    root = args.bundle.resolve()
    if digest(root / "MANIFEST.json") != args.manifest_sha256:
        raise ValueError("Staged manifest SHA256 changed")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest["run_id"] != "public-d4-preflight-v1" or manifest["authorized_for_submission"]:
        raise ValueError("Unexpected preflight scope")
    for name, record in manifest["files"].items():
        path = root / name
        if path.parent != root or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Preflight input changed: {name}")
    helper = load_module(root / "public_d4_preflight.py", "public_d4_preflight_runtime")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout
    helper.verify_idle_gpu_query(processes)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      BIOHUB_EDGE_FEATURE_TTA="1", BIOHUB_SECONDARY_EDGE_FEATURE_TTA="1",
                      BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT="0.75", BIOHUB_DEEPCENTER_TTA="1",
                      BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION="0.90")
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one freed, visible CUDA GPU is required")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    result.update(torch_version=torch.__version__, numpy_version=np.__version__,
                  device=torch.cuda.get_device_name(0), gpu_idle_at_launch=True,
                  manifest_sha256=args.manifest_sha256, arms={})
    temporal = load_module(root / "temporal_unet.py", "d4_public_temporal")
    transformer = load_module(root / "simple_node_transformer.py", "d4_public_transformer")
    namespace = dict(torch=torch, nn=torch.nn, SimpleNodeTransformer=transformer.SimpleNodeTransformer)
    definitions = helper.named_definitions((root / "train_unet_transformer.py").read_text(), ("UNetNodeTransformer",))
    exec(compile(definitions, "reviewed-public-model-class", "exec"), namespace)
    models = []
    for name in ("primary", "secondary"):
        state = torch.load(root / f"{name}.pth", map_location="cpu", weights_only=True)
        model = namespace["UNetNodeTransformer"](
            temporal.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128]),
            unet_out_channels=32, pos_feat_dim=32,
        )
        # The public inference artifact is the bare best-state dictionary,
        # not the separate larger optimizer/resume checkpoint_last.pth.
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).eval())
        del state
    with np.load(root / "frames.npz", allow_pickle=False) as archive:
        if archive.files != ["raw"]:
            raise ValueError("Image-only NPZ required")
        raw = archive["raw"]
    if list(raw.shape) != manifest["raw_shape"] or raw.dtype != np.uint16:
        raise ValueError("Real full-frame image contract changed")
    q = manifest["quantiles"]
    image = np.maximum((raw[:, :, ::4, ::4].astype(np.float32) - q["0.001"])
                       / (q["0.999"] - q["0.001"] + 1e-6), 0)
    images = torch.from_numpy(image[None]).to(device)

    class CountedEncoder:
        def __init__(self, model):
            self.model, self.inputs = model, []

        def encode(self, value):
            # Input fingerprints verify the actual executed views, not a label
            # on a wrapper. This costs CPU copies only in this tiny preflight.
            data = value.detach().cpu().contiguous().numpy()
            self.inputs.append((tuple(data.shape), hashlib.sha256(data.tobytes()).hexdigest()))
            return self.model.encode(value)

    saved = {}
    with torch.inference_mode():
        for arm, filename, expected_unique in (
            ("original", "public-predictor-original.py", 7),
            ("corrected", "public-predictor-d4-corrected.py", 8),
        ):
            source = (root / filename).read_text()
            env = dict(torch=torch, np=np, F=F, os=os,
                       cfg=SimpleNamespace(det_tta=True, det_threshold=0.965), pool_k=(3, 3, 3))
            exec(compile(helper.named_definitions(source, ("_detect_cells_pooled",)), "reviewed-public-peaks", "exec"), env)
            exec(compile(helper.encoder_only_function(source), "reviewed-public-encoder-only", "exec"), env)
            wrapped = [CountedEncoder(model) for model in models]
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            features, secondary, detections = env["public_encoder_pass"](images, *wrapped)
            torch.cuda.synchronize()
            tensors = [features, secondary, *detections]
            if any(not torch.isfinite(t).all() for t in tensors):
                raise ValueError("Nonfinite real encoder output")
            counts = [dict(calls=len(m.inputs), unique_views=len(set(m.inputs))) for m in wrapped]
            if any(r != dict(calls=8, unique_views=expected_unique) for r in counts):
                raise ValueError(f"Actual encoder views changed: {counts}")
            saved[arm] = [t.detach().cpu().numpy() for t in tensors]
            result["arms"][arm] = dict(encoder_views=counts, shapes=[list(t.shape) for t in tensors],
                                       seconds=time.monotonic()-started,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated())
            del features, secondary, detections, tensors, wrapped
        result["encoder_mean_absolute_deltas"] = [float(np.mean(np.abs(a-b)))
            for a, b in zip(saved["original"], saved["corrected"])]
        if any(v <= 0 or not np.isfinite(v) for v in result["encoder_mean_absolute_deltas"]):
            raise ValueError("Real correction did not affect every encoder output")
        np.savez_compressed(args.output / "paired-encoder-outputs.npz",
                            **{f"{arm}_{i}": value for arm, rows in saved.items() for i, value in enumerate(rows)})
        del saved, models, model, images
        torch.cuda.empty_cache()

        dc_source = (root / "public-postprocess-d4-corrected.py").read_text()
        env = dict(torch=torch, np=np, os=os, DEEPCENTER_SCORE_CACHE_MAX_FRAMES=8,
                   read_test_frame=lambda dataset, t, cache: raw[t])
        names = ("_DCConvBlock3d", "_DCDeepCenterUNet3D", "_dc_pool_frame_xy",
                 "_dc_normalize_dynamic_range", "_dc_cache_trim", "deepcenter_heatmap_for_frame")
        exec(compile(helper.named_definitions(dc_source, names), "reviewed-public-deepcenter", "exec"), env)
        state = torch.load(root / "deepcenter.pt", map_location="cpu", weights_only=True)
        if state["epoch"] != 2:
            raise ValueError("Pinned epoch-2 DeepCenter required")
        cfg = SimpleNamespace(**state["config"])
        dc_model = env["_DCDeepCenterUNet3D"](base_channels=cfg.base_channels)
        dc_model.load_state_dict(state["model_state"], strict=True)
        dc_model.to(device).eval()
        del state

        class CountedCenter:
            def __init__(self):
                self.inputs = []

            def __call__(self, value):
                data = value.detach().cpu().contiguous().numpy()
                self.inputs.append((data.shape, hashlib.sha256(data.tobytes()).hexdigest()))
                return dc_model(value)

        dc_outputs = {}
        for arm, rotation, unique in (("original", 1, 7), ("corrected", 2, 8)):
            code = helper.named_definitions(dc_source, ("deepcenter_heatmap_for_frame",))
            if rotation == 1:
                old = "torch_mod.rot90(tensor, 2, dims=(-2, -1))"
                inverse = "torch_mod.rot90(model(at).transpose(-1, -2), -2, dims=(-2, -1))"
                if code.count(old) != 1 or code.count(inverse) != 1:
                    raise ValueError("DeepCenter controlled geometry reversal drift")
                code = code.replace(old, old.replace(", 2,", ", 1,"), 1)
                code = code.replace(inverse, inverse.replace(", -2, dims", ", -1, dims"), 1)
            exec(compile(code, "paired-deepcenter-geometry", "exec"), env)
            counted = CountedCenter()
            bundle = dict(model=counted, cfg=cfg, device=device, torch=torch)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            heatmap = env["deepcenter_heatmap_for_frame"](manifest["movie"], 0, bundle, {}, {})
            torch.cuda.synchronize()
            if not np.isfinite(heatmap).all() or len(counted.inputs) != 8 or len(set(counted.inputs)) != unique:
                raise ValueError("Actual DeepCenter views/output failed")
            dc_outputs[arm] = heatmap
            result["arms"][arm]["deepcenter"] = dict(calls=8, unique_views=unique,
                seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        delta = float(np.mean(np.abs(dc_outputs["original"]-dc_outputs["corrected"])))
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("No actual DeepCenter change")
        result["deepcenter_mean_absolute_delta"] = delta
        np.savez_compressed(args.output / "paired-deepcenter-outputs.npz", **dc_outputs)
    # Weights and inputs must be unchanged; hashes cover the actual staged files.
    if any(digest(root/name) != row["sha256"] for name, row in manifest["files"].items()):
        raise ValueError("Preflight modified a source or input")
    result.update(status="functionality_passed", inputs_unchanged=True,
                  quality_gain_established=False, independently_held_out=False,
                  model_training=False, ground_truth_opened=False,
                  competition_submission_performed=False, authorized_for_submission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(run_id="public-d4-preflight-v1", status="running", authorized_for_submission=False)

    def timeout():
        (args.output / "timeout.json").write_text(json.dumps(dict(status="timeout", wall_cap_seconds=1200)))
        os._exit(124)  # only this owned preflight process, never another project

    timer = threading.Timer(1200, timeout)
    timer.daemon = True
    timer.start()
    try:
        run(args, result)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        timer.cancel()
        result["elapsed_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
