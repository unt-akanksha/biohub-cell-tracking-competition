"""Tiny source-movie reload/inference/serialization/scoring gate."""
import hashlib
import time


def run_smoke(checkpoint, movie, output, device='cuda', encode_patch=None,
              pre_motion_patch=None,standalone_image_flow=False,flow_patch=None,
              skip_zero_neural=False):
    import numpy as np
    import polars as pl
    import torch
    import tracksdata as td
    from train_unet_transformer import UNetNodeTransformer, TemporalUNet3D
    from predict_unet_transformer import predict_video, PredictConfig
    from tracking_cellmot.io import open_dataset
    from tracking_cellmot import metrics
    from independent_real_baseline import install_empty_attention_guard
    from empty_graph_schema import restore_empty_spatial_schema
    started = time.monotonic()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if movie.name not in state['identity']['training_stems']:
        raise ValueError('GPU smoke may open only a recorded training movie')
    devices = list(range(torch.cuda.device_count())) if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model = UNetNodeTransformer(TemporalUNet3D(in_channels=1, out_channels=32,
            layers=[32,64,128]), unet_out_channels=32, pos_feat_dim=32).to(device)
        model.load_state_dict(state['model'], strict=True)
        install_empty_attention_guard(model)
        model.eval()
        pre_motion_receipt = pre_motion_patch(model) if pre_motion_patch is not None else None
        flow_receipt = None
        if flow_patch is not None and not state['identity'].get('image_motion'):
            raise ValueError('Flow patch requires a verified embedded image-motion checkpoint')
        if standalone_image_flow and not state['identity'].get('image_motion'):
            raise ValueError('Standalone flow requires an embedded image-motion checkpoint')
        if state['identity'].get('motion_residual'):
            from motion_residual import contract,install_motion_residual
            if state['identity']['motion_residual'] != contract():
                raise ValueError('Motion residual inference contract mismatch')
            install_motion_residual(model,inference=not bool(state['identity'].get('image_motion')))
        if state['identity'].get('image_motion'):
            from image_motion_residual import embedded_flow,install_image_motion_residual
            flow_model = embedded_flow(state,device)
            flow_receipt = flow_patch(flow_model) if flow_patch is not None else None
            install_image_motion_residual(model,flow_model,inference=True,
                calibration=[0.,1.,-4.5] if standalone_image_flow else None,
                skip_zero_neural=skip_zero_neural)
        model.eval()
        encode_receipt = encode_patch(model) if encode_patch is not None else None
        cfg = PredictConfig(det_threshold=float(torch.sigmoid(torch.tensor(.3))),
            det_tta=False, pool_kernel_um=5., max_parents_per_node=1, max_children_per_node=2)
        if state['identity'].get('motion_residual'):
            cfg.edge_activation = 'sigmoid'
        with torch.no_grad():
            coords, edges = predict_video(model, movie, torch.device(device), cfg,
                window_size=2, max_frames=3, unet_batch_size=1, downsample=(1,4,4))
        if not np.isfinite(coords).all() or (coords < 0).any():
            raise ValueError('Invalid smoke coordinates')
        graph = td.graph.InMemoryGraph()
        for axis in ('z','y','x'):
            graph.add_node_attr_key(axis, pl.Float64, 0.)
        ids = graph.bulk_add_nodes([dict(t=int(c[0]),z=float(c[1]),y=float(c[2]),x=float(c[3])) for c in coords])
        pairs = []
        for source, target, probability, distance in edges:
            if (not np.isfinite([probability,distance]).all()
                    or not 0 <= source < len(coords) or not 0 <= target < len(coords)
                    or coords[target,0] != coords[source,0]+1):
                raise ValueError('Invalid smoke edge')
            pairs.append(dict(source_id=ids[source],target_id=ids[target]))
        if pairs:
            graph.bulk_add_edges(pairs)
        path = output/'gpu_smoke.geff'
        graph.to_geff(path)
        restored = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        restore_empty_spatial_schema(restored)
        if restored.num_nodes() != len(coords) or restored.num_edges() != len(edges):
            raise ValueError('Smoke GEFF round-trip mismatch')
        truth = open_dataset(movie, require_tracks=True, load_image=False).tracks
        truth = truth.filter(td.NodeAttr('t') < 3).subgraph()
        restore_empty_spatial_schema(truth)
        scored = metrics.evaluate(restored, truth, scale=(1.625,.40625,.40625), max_distance=7.)
    return dict(status='passed', device=device, checkpoint_step=state['step'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        movie=movie.name, frames=3, strict_reload=True, geff_round_trip=True,
        predicted_nodes=len(coords), predicted_edges=len(edges), scorer_counts=scored._asdict(),
        elapsed_seconds=time.monotonic()-started, scope='Training-movie functionality only',
        authorized_for_submission=False,
        encode_patch_receipt=encode_receipt,pre_motion_patch_receipt=pre_motion_receipt,
        flow_patch_receipt=flow_receipt,
        motion_execution_receipt=getattr(model,'_image_motion_execution',None),
        standalone_image_flow=standalone_image_flow)
