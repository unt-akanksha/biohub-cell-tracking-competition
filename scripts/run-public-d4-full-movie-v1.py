"""Frozen, label-free sequential full-pipeline smoke/full diagnostic on Antelume."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(args, result):
    helper = load_module(args.bundle / 'public_d4_full_movie.py', 'd4_full_helper')
    pre = load_module(args.bundle / 'public_d4_preflight.py', 'd4_preflight_helper')
    contract = json.loads(args.contract.read_text())
    if helper.sha(args.contract) != args.contract_sha256:
        raise ValueError('Frozen run contract changed')
    if args.mode == 'full':
        if args.smoke_proof is None:
            raise ValueError('End-to-end smoke proof required before full run')
        smoke = json.loads(args.smoke_proof.read_text())
        if (smoke.get('status') != 'functionality_passed' or smoke.get('mode') != 'smoke'
                or smoke.get('contract_sha256') != args.contract_sha256
                or smoke.get('inputs_unchanged') is not True):
            raise ValueError('Same-contract successful smoke required')
    for name, digest in contract['bundle_sha256'].items():
        if Path(name).name != name or helper.sha(args.bundle / name) != digest:
            raise ValueError(f'Runtime input changed: {name}')
    if contract['stems'] != list(helper.STEMS) or contract['authorized_for_submission']:
        raise ValueError('Unexpected diagnostic scope')
    image_manifest = json.loads((args.images / 'IMAGE_MANIFEST.json').read_text())
    if (helper.sha(args.images / 'IMAGE_MANIFEST.json') != contract['image_manifest_sha256']
            or image_manifest['status'] != 'complete' or image_manifest['stems'] != list(helper.STEMS)
            or len(image_manifest['records']) != 408):
        raise ValueError('Incomplete image inventory')
    for row in image_manifest['records']:
        path = args.images / row['path']
        if path.stat().st_size != row['bytes'] or helper.sha(path) != row['sha256']:
            raise ValueError('Image checksum mismatch')
    pre.verify_idle_gpu_query(subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15).stdout)
    frozen = json.loads((args.bundle / 'resolved-public-config.json').read_text())
    for key in list(os.environ):
        if key.startswith('BIOHUB_'):
            del os.environ[key]
    os.environ.update(frozen['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      POLARS_MAX_THREADS='2')
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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One freed Antelume CUDA device required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1729)
    device = torch.device('cuda:0')
    temporal = load_module(args.bundle / 'temporal_unet.py', 'd4_temporal')
    transformer = load_module(args.bundle / 'simple_node_transformer.py', 'd4_transformer')
    model_env = dict(torch=torch, nn=torch.nn, np=np, _POS_EMBED_DIM=8,
                     SimpleNodeTransformer=transformer.SimpleNodeTransformer)
    exec(compile(pre.named_definitions((args.bundle / 'train_unet_transformer.py').read_text(),
                 ('UNetNodeTransformer', 'extract_pos_features')), 'pinned-model-definitions', 'exec'), model_env)
    models = []
    for name in ('primary', 'secondary'):
        model = model_env['UNetNodeTransformer'](
            temporal.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128]),
            unet_out_channels=32, pos_feat_dim=32)
        model.load_state_dict(torch.load(args.bundle / f'{name}.pth', map_location='cpu', weights_only=True), strict=True)
        models.append(model.to(device).eval())
    post_source = (args.bundle / 'public-postprocess-d4-corrected.py').read_text()
    dc_env = dict(torch=torch)
    exec(compile(pre.named_definitions(post_source, ('_DCConvBlock3d', '_DCDeepCenterUNet3D')),
                 'pinned-deepcenter-model', 'exec'), dc_env)
    state = torch.load(args.bundle / 'deepcenter.pt', map_location='cpu', weights_only=True)
    if state['epoch'] != 2:
        raise ValueError('Expected exact epoch-2 checkpoint')
    dc_config = SimpleNamespace(**state['config'])
    dc_model = dc_env['_DCDeepCenterUNet3D'](base_channels=dc_config.base_channels)
    dc_model.load_state_dict(state['model_state'], strict=True)
    dc_bundle = dict(model=dc_model.to(device).eval(), cfg=dc_config, device=device, torch=torch)
    del state
    result.update(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  contract_sha256=args.contract_sha256,
                  numpy_version=np.__version__, mode=args.mode, movies={}, ground_truth_opened=False,
                  independently_held_out=False, kaggle_gpu_hours=0)
    frames = 8 if args.mode == 'smoke' else 100
    stems = helper.STEMS[:1] if args.mode == 'smoke' else helper.STEMS
    for stem in stems:
        result['movies'][stem] = {}
        for arm in ('original', 'corrected'):
            started = time.monotonic()
            out = args.output / f'{stem}-{arm}'
            out.mkdir()
            module = ModuleType('d4_public_' + arm)
            sys.modules[module.__name__] = module
            env = module.__dict__
            env.update(frozen['globals'])
            env.update(torch=torch, np=np, F=F, os=os, Path=Path, json=json, math=math,
                       td=td, pl=pl, zarr=zarr, tqdm=tqdm, dataclass=dataclass, blosc2=blosc2,
                       cKDTree=cKDTree, linear_sum_assignment=linear_sum_assignment,
                       _POS_EMBED_DIM=8, INTERACTIVE=False, _RUN_OUTPUT=out,
                       extract_pos_features=model_env['extract_pos_features'],
                       open_dataset=helper.image_metadata, TEST_DIR=args.images / 'train',
                       VOXEL_SCALE_UM=(1.625, .40625, .40625))
            filename = 'public-predictor-original.py' if arm == 'original' else 'public-predictor-d4-corrected.py'
            source = helper.adapt_predictor_logs((args.bundle / filename).read_text())
            exec(compile(pre.named_definitions(source, helper.PREDICTOR_NAMES), 'pinned-public-predictor', 'exec'), env)
            arm_post = helper.original_postprocess(post_source) if arm == 'original' else post_source
            exec(compile(pre.named_definitions(arm_post, helper.POST_NAMES), 'pinned-public-postprocess', 'exec'), env)
            cfg = env['PredictConfig'](det_threshold=env['DET_THRESHOLD'], threshold=.48,
                    use_ilp=env['USE_ILP'], ilp_edge_weight=env['ILP_EDGE_WEIGHT'],
                    ilp_appearance_weight=env['ILP_APPEARANCE_WEIGHT'],
                    ilp_disappearance_weight=env['ILP_DISAPPEARANCE_WEIGHT'],
                    ilp_division_weight=env['ILP_DIVISION_WEIGHT'])
            print(json.dumps(dict(event='movie_started', stem=stem, arm=arm, frames=frames)), flush=True)
            torch.cuda.reset_peak_memory_stats()
            with (out / 'inference.log').open('x') as log, contextlib.redirect_stdout(log), torch.inference_mode():
                coords, edges = env['predict_video'](models[0], args.images / 'train' / (stem+'.zarr'),
                    device, cfg, window_size=2, max_frames=frames, unet_batch_size=env['UNET_BATCH_SIZE'],
                    downsample=(1, 4, 4), secondary_model=models[1], secondary_edge_weight=.15,
                    secondary_detection_weight=.80, secondary_link_mode='low_margin_consensus',
                    secondary_mix_temperature=1., secondary_low_margin_max=.35)
                torch.cuda.synchronize()
                inference_seconds = time.monotonic()-started
                if set(coords[:, 0].tolist()) != set(range(frames)) or not edges:
                    raise ValueError('Incomplete detection/pair coverage')
                np.savez_compressed(out / 'raw-candidates.npz', coords=coords, edges=np.asarray(edges))
                graph = env['build_graph'](coords, edges)
                solver = td.solvers.ILPSolver(edge_weight=cfg.ilp_edge_weight * td.EdgeAttr('edge_prob'),
                    appearance_weight=cfg.ilp_appearance_weight,
                    disappearance_weight=cfg.ilp_disappearance_weight, division_weight=cfg.ilp_division_weight)
                graph = solver.solve(graph)
                if not graph.num_nodes() or not graph.num_edges():
                    raise ValueError('ILP produced an empty graph')
                nodes = {int(r['node_id']): {k: int(r[k]) if k in ('node_id','t') else float(r[k])
                         for k in ('node_id','t','z','y','x')} for r in graph.node_attrs().iter_rows(named=True)}
                graph_edges = [dict(source_id=int(r['source_id']), target_id=int(r['target_id']),
                                    edge_prob=float(r['edge_prob'])) for r in graph.edge_attrs().iter_rows(named=True)]
                (out / 'pre-postprocess.json').write_text(json.dumps(dict(nodes=nodes, edges=graph_edges)))
                nodes, graph_edges, stats = env['filter_output_graph'](nodes, graph_edges,
                                                dataset=stem, deepcenter_bundle=dc_bundle)
                payload = helper.csv_equivalent_graph(nodes, graph_edges, frames)
                (out / 'prediction.json').write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
                (out / 'postprocess-stats.json').write_text(json.dumps(stats, sort_keys=True, allow_nan=False))
                torch.cuda.synchronize()
            row = dict(seconds=time.monotonic()-started, inference_seconds=inference_seconds,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(), frames=frames,
                       nodes=len(payload['nodes']), edges=len(payload['edges']),
                       prediction_sha256=helper.sha(out / 'prediction.json'))
            result['movies'][stem][arm] = row
            (args.output / 'prelabel-progress.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(event='movie_completed', stem=stem, arm=arm, **row)), flush=True)
            del coords, edges, graph, nodes, graph_edges, payload, env, module
            gc.collect()
            torch.cuda.empty_cache()
    if any(helper.sha(args.bundle/name) != value for name, value in contract['bundle_sha256'].items()):
        raise ValueError('Frozen runtime changed during run')
    result.update(status='functionality_passed' if args.mode=='smoke' else 'complete_prelabel_predictions',
                  quality_gain_established=False, authorized_for_submission=False, inputs_unchanged=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle','images','contract','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('--mode', choices=('smoke','full'), required=True)
    p.add_argument('--smoke-proof', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cap = 600 if args.mode == 'smoke' else 3600
    result = dict(run_id='public-d4-full-movie-v1', status='running', authorized_for_submission=False)
    def timeout():
        (args.output / 'timeout.json').write_text(json.dumps(dict(status='timeout', wall_cap_seconds=cap)))
        os._exit(124)
    timer = threading.Timer(cap, timeout); timer.daemon=True; timer.start()
    try:
        execute(args, result)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        timer.cancel()
        result['elapsed_seconds'] = time.monotonic()-started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
