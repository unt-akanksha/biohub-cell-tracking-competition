#!/usr/bin/env python
"""Faithful post-process configuration sweep on Antelume. Runs ON the instance.

Why this exists: the published notebook chooses its post-process configuration
with an in-kernel sweep over 8 held-out TRAIN movies, because the validator and
sweep are charged against the competition's 12-hour cap. That single step is
worth about +0.009 on the leaderboard -- an order of magnitude more than anything
this project has measured elsewhere -- yet it rests on 8 of 199 labelled movies.

An offline screen was tried first and failed its own fidelity test: with the
DeepCenter veto and synthetic gap refinement disabled for want of the images, the
base configuration moved by +0.00259 pooled, and by +0.045 on one movie, against
a signal of interest of about 0.002. Here the images are resident per movie, so
the veto runs live and `filter_output_graph` is the real one.

Per movie: predict, solve, then post-process and score once per candidate
configuration against local ground truth with the patched official scorer.
Images are deleted as soon as a movie is finished, so peak disk stays near one
movie regardless of how many are swept.

Nothing is selected here. This emits per-movie, per-configuration rows; the
choice, including holding an embryo out, is made afterwards from the table.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
MAX_DISTANCE_UM = 7.0
FRAMES = 100


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scorer(bundle: Path):
    folder = bundle / "scorer"
    expected = {
        "metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
        "division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
    }
    for name, digest in expected.items():
        payload = (folder / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Patched official scorer changed: {name}")
    name = "_antelume_scorer"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    return importlib.import_module(name + ".metrics")


def build_env(bundle: Path, helper, pre, frozen_globals, overrides, images, extras):
    """A fresh module namespace with the pinned predictor+postprocess installed."""
    import blosc2
    import numpy as np
    import polars as pl
    import torch
    import torch.nn.functional as F
    import tracksdata as td
    import zarr
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from tqdm import tqdm

    module = ModuleType("antelume_sweep_%d" % next(_COUNTER))
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update(frozen_globals)
    if overrides:
        unknown = set(overrides) - set(frozen_globals)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        env.update(overrides)
    env.update(
        torch=torch, np=np, F=F, os=os, Path=Path, json=json, math=math,
        td=td, pl=pl, zarr=zarr, tqdm=tqdm, dataclass=dataclass, blosc2=blosc2,
        cKDTree=cKDTree, linear_sum_assignment=linear_sum_assignment,
        _POS_EMBED_DIM=8, INTERACTIVE=False,
        open_dataset=helper.image_metadata,
        TEST_DIR=images / "train",
        VOXEL_SCALE_UM=VOXEL_SCALE_UM,
        **extras,
    )
    return env, module


def _counter():
    value = 0
    while True:
        yield value
        value += 1


_COUNTER = _counter()


def score_rows(scorer, nodes, edges, stem, truth_root):
    import tracksdata as td
    from geff import GeffMetadata
    import polars as pl

    truth_path = truth_root / f"{stem}.geff"
    truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
    graph = td.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    ids = sorted(nodes, key=int)
    mapped = graph.bulk_add_nodes(
        [{"t": int(nodes[i]["t"]), "z": float(nodes[i]["z"]),
          "y": float(nodes[i]["y"]), "x": float(nodes[i]["x"])} for i in ids]
    )
    mapping = dict(zip((int(i) for i in ids), mapped))
    graph.bulk_add_edges(
        [dict(source_id=mapping[int(e["source_id"])], target_id=mapping[int(e["target_id"])])
         for e in edges]
    )
    result = scorer.evaluate(graph, truth, scale=VOXEL_SCALE_UM, max_distance=MAX_DISTANCE_UM)
    count = float(GeffMetadata.read(str(truth_path)).extra["estimated_number_of_nodes"])
    row = dict(scorer.per_sample_metrics(result, count, scorer.node_recall(graph, truth)))
    row["stem"] = stem
    row["embryo"] = stem.split("_")[0]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--stems", type=Path, required=True, help="JSON list")
    parser.add_argument("--candidates", type=Path, required=True, help="JSON {label: overrides}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--keep-images", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.bundle))
    import numpy as np
    import torch
    import tracksdata as td

    import public_d4_full_movie as helper
    import public_d4_preflight as pre

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("A CUDA device is required")

    frozen = json.loads((args.bundle / "resolved-public-config.json").read_text())["globals"]
    stems = json.loads(args.stems.read_text())
    candidates = json.loads(args.candidates.read_text())
    if "base" in candidates:
        raise SystemExit("'base' is implicit and must not appear in the candidate table")
    scorer = load_scorer(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)

    # Deterministic settings, matching the 2026-09-10 Antelume run exactly.
    for key in [k for k in os.environ if k.startswith("BIOHUB_")]:
        del os.environ[key]
    os.environ.update(json.loads((args.bundle / "resolved-public-config.json").read_text())["environment"])
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      POLARS_MAX_THREADS="2")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.70)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1729)

    post_source_raw = (args.bundle / "public-postprocess-d4-corrected.py").read_text()

    # Models, loaded once.
    temporal = _load_module(args.bundle / "temporal_unet.py", "d4_temporal")
    transformer = _load_module(args.bundle / "simple_node_transformer.py", "d4_transformer")
    model_env = dict(torch=torch, nn=torch.nn, np=np, _POS_EMBED_DIM=8,
                     SimpleNodeTransformer=transformer.SimpleNodeTransformer)
    exec(compile(pre.named_definitions((args.bundle / "train_unet_transformer.py").read_text(),
                 ("UNetNodeTransformer", "extract_pos_features")),
                 "pinned-model-definitions", "exec"), model_env)
    models = []
    for name in ("primary", "secondary"):
        model = model_env["UNetNodeTransformer"](
            temporal.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128]),
            unet_out_channels=32, pos_feat_dim=32)
        model.load_state_dict(
            torch.load(args.bundle / f"{name}.pth", map_location="cpu", weights_only=True),
            strict=True,
        )
        models.append(model.to(device).eval())

    dc_env = dict(torch=torch)
    exec(compile(pre.named_definitions(post_source_raw, ("_DCConvBlock3d", "_DCDeepCenterUNet3D")),
                 "pinned-deepcenter-model", "exec"), dc_env)
    state = torch.load(args.bundle / "deepcenter.pt", map_location="cpu", weights_only=True)
    if state["epoch"] != 2:
        raise SystemExit("Expected the exact epoch-2 DeepCenter checkpoint")
    dc_config = SimpleNamespace(**state["config"])
    dc_model = dc_env["_DCDeepCenterUNet3D"](base_channels=dc_config.base_channels)
    dc_model.load_state_dict(state["model_state"], strict=True)
    dc_bundle = dict(model=dc_model.to(device).eval(), cfg=dc_config, device=device, torch=torch)
    del state

    predictor_source = helper.adapt_predictor_logs(
        (args.bundle / "public-predictor-original.py").read_text()
    )
    # Stock seven-view geometry: the complete-D4 correction was rejected on the
    # leaderboard at 0.940 against 0.947, so the front end stays as published.
    post_source = helper.original_postprocess(post_source_raw)

    all_rows: list[dict] = []
    manifest: dict = {"run_id": "antelume-config-sweep-v1", "stems": stems,
                      "candidates": candidates, "frames": args.frames, "movies": {},
                      "selection_performed": False, "ground_truth_opened": True,
                      "device": torch.cuda.get_device_name()}

    for index, stem in enumerate(stems, start=1):
        movie_started = time.monotonic()
        zarr_path = args.images / "train" / f"{stem}.zarr"
        if not zarr_path.exists():
            print(json.dumps(dict(event="missing_images", stem=stem)), flush=True)
            continue

        env, module = build_env(args.bundle, helper, pre, frozen, None, args.images,
                                dict(extract_pos_features=model_env["extract_pos_features"],
                                     _RUN_OUTPUT=args.output))
        exec(compile(pre.named_definitions(predictor_source, helper.PREDICTOR_NAMES),
                     "pinned-predictor", "exec"), env)
        cfg = env["PredictConfig"](
            det_threshold=env["DET_THRESHOLD"], threshold=0.48, use_ilp=env["USE_ILP"],
            ilp_edge_weight=env["ILP_EDGE_WEIGHT"],
            ilp_appearance_weight=env["ILP_APPEARANCE_WEIGHT"],
            ilp_disappearance_weight=env["ILP_DISAPPEARANCE_WEIGHT"],
            ilp_division_weight=env["ILP_DIVISION_WEIGHT"],
        )
        print(json.dumps(dict(event="predict_started", stem=stem, index=index,
                              total=len(stems))), flush=True)
        torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), torch.inference_mode():
            coords, edges = env["predict_video"](
                models[0], zarr_path, device, cfg, window_size=2, max_frames=args.frames,
                unet_batch_size=env["UNET_BATCH_SIZE"], downsample=(1, 4, 4),
                secondary_model=models[1], secondary_edge_weight=0.15,
                secondary_detection_weight=0.80, secondary_link_mode="low_margin_consensus",
                secondary_mix_temperature=1.0, secondary_low_margin_max=0.35,
            )
            torch.cuda.synchronize()
            if set(coords[:, 0].tolist()) != set(range(args.frames)) or not edges:
                raise ValueError(f"Incomplete detection coverage for {stem}")
            graph = env["build_graph"](coords, edges)
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            graph = solver.solve(graph)
            raw_nodes = {int(r["node_id"]): {k: int(r[k]) if k in ("node_id", "t") else float(r[k])
                         for k in ("node_id", "t", "z", "y", "x")}
                         for r in graph.node_attrs().iter_rows(named=True)}
            raw_edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]),
                              edge_prob=float(r["edge_prob"]))
                         for r in graph.edge_attrs().iter_rows(named=True)]
        predict_seconds = time.monotonic() - movie_started
        (args.output / f"{stem}-raw.json").write_text(
            json.dumps(dict(nodes=raw_nodes, edges=raw_edges), sort_keys=True, allow_nan=False)
        )
        del coords, edges, graph, env, module

        movie_rows = []
        for label, overrides in [("base", None), *candidates.items()]:
            started = time.monotonic()
            penv, pmodule = build_env(args.bundle, helper, pre, frozen, overrides, args.images,
                                      dict(_RUN_OUTPUT=args.output))
            exec(compile(pre.named_definitions(post_source, helper.POST_NAMES),
                         "pinned-postprocess", "exec"), penv)
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                nodes, processed_edges, stats = penv["filter_output_graph"](
                    copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges),
                    dataset=stem, deepcenter_bundle=dc_bundle,
                )
            row = score_rows(scorer, nodes, processed_edges, stem, args.truth)
            row["config"] = label
            row["seconds"] = time.monotonic() - started
            row["nodes"] = len(nodes)
            row["edges"] = len(processed_edges)
            movie_rows.append(row)
            all_rows.append(row)
            print(json.dumps(dict(event="config_scored", stem=stem, config=label,
                                  seconds=round(row["seconds"], 1))), flush=True)
            del penv, pmodule, nodes, processed_edges

        manifest["movies"][stem] = dict(
            predict_seconds=predict_seconds,
            total_seconds=time.monotonic() - movie_started,
            peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),
            raw_nodes=len(raw_nodes), raw_edges=len(raw_edges),
            configs=len(movie_rows),
        )
        with (args.output / "rows.jsonl").open("a") as handle:
            for row in movie_rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        (args.output / "progress.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(json.dumps(dict(event="movie_completed", stem=stem, index=index,
                              seconds=round(time.monotonic() - movie_started, 1))), flush=True)

        if not args.keep_images:
            shutil.rmtree(zarr_path, ignore_errors=True)
        del raw_nodes, raw_edges

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(dict(event="sweep_complete", movies=len(manifest["movies"]),
                          rows=len(all_rows))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
