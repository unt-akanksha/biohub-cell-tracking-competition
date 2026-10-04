"""CPU checkpoint/inference/GEFF/scorer smoke on an artificial three-frame movie.

This is a functionality check only. Synthetic scores cannot promote a model.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(root, notebook):
    verifier = runpy.run_path(str(ROOT/'scripts/verify-independent-real-pilot.py'))
    verified = verifier['verify'](root, notebook)
    import numpy as np
    import torch
    import tracksdata as td
    import zarr
    torch.set_num_threads(2)
    vendor = ROOT/'.biohub/vendor/kaggle-cell-tracking-competition'
    sys.path[:0] = [str(vendor/'scripts'), str(vendor/'src'), str(ROOT)]
    from train_unet_transformer import UNetNodeTransformer, TemporalUNet3D
    from predict_unet_transformer import predict_video, PredictConfig
    from research.independent_real_baseline import install_empty_attention_guard
    from research.empty_graph_schema import restore_empty_spatial_schema
    replay = runpy.run_path(str(ROOT/'scripts/replay-focus-bridge-official.py'))
    state = torch.load(root/'outputs/last.pt', map_location='cpu', weights_only=True)
    model = UNetNodeTransformer(TemporalUNet3D(in_channels=1, out_channels=32,
        layers=[32,64,128]), unet_out_channels=32, pos_feat_dim=32)
    model.load_state_dict(state['model'], strict=True)
    install_empty_attention_guard(model)
    model.eval()
    with tempfile.TemporaryDirectory(prefix='biohub-smoke-') as temp:
        directory = Path(temp)
        grid = np.indices((16,64,64), dtype=np.float32)
        frames, truth_nodes, truth_edges = [], {}, []
        for t in range(3):
            volume = np.zeros((16,64,64), dtype=np.float32)
            for j, center in enumerate(((8,24+t,32), (8,40+t,20))):
                distance = sum(((grid[a]-center[a])/(1.5 if a == 0 else 3.))**2 for a in range(3))
                volume += np.exp(-.5*distance)
                node = t*2+j
                truth_nodes[str(node)] = dict(t=t,z=center[0],y=center[1],x=center[2])
                if t:
                    truth_edges.append(dict(source_id=node-2, target_id=node))
            frames.append(volume)
        image = zarr.open_group(str(directory/'fixture.zarr'), mode='w')
        image.create_array('0', data=np.stack(frames), chunks=(1,16,64,64))
        image.attrs['image_statistics'] = {'quantiles':{'0.001':0.,'0.999':1.}}
        config = PredictConfig(det_threshold=float(torch.sigmoid(torch.tensor(.3))),
            det_tta=False, pool_kernel_um=5., max_parents_per_node=1, max_children_per_node=2)
        with torch.no_grad():
            coords, edges = predict_video(model, directory/'fixture', torch.device('cpu'),
                config, window_size=2, unet_batch_size=1, downsample=(1,4,4))
        if not np.isfinite(coords).all():
            raise ValueError('Nonfinite inference coordinates')
        payload = dict(nodes={str(i):dict(zip(('t','z','y','x'), map(float,c))) for i,c in enumerate(coords)},
            edges=[dict(source_id=int(e[0]), target_id=int(e[1])) for e in edges])
        predicted = replay['prediction_graph'](payload)
        path = directory/'prediction.geff'
        predicted.to_geff(path)
        restored = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        restored_schema = restore_empty_spatial_schema(restored)
        if restored.num_nodes() != len(coords) or restored.num_edges() != len(edges):
            raise ValueError('GEFF round trip changed graph size')
        truth = replay['prediction_graph'](dict(nodes=truth_nodes, edges=truth_edges))
        result = replay['load_scorer']('current').evaluate(restored, truth,
            scale=(1.625,.40625,.40625), max_distance=7.)
        return dict(status='cpu_functionality_smoke_passed', checkpoint_sha256=verified['checkpoint_sha256'],
            strict_reload=True, inference_frames=3, predicted_nodes=len(coords), predicted_edges=len(edges),
            geff_round_trip=True, organizer_scorer_executed=True, scorer_counts=result._asdict(),
            empty_schema_restored=restored_schema,
            data_scope='Artificial fixture, not competition data',
            coverage='CPU reload/inference/serialization/scoring; does not prove GPU inference or complete-movie accuracy',
            authorized_for_submission=False, larger_training_authorized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--notebook', type=Path, default=ROOT/'kaggle/biohub-independent-real-pilot-v1/biohub-independent-real-pilot-v1.ipynb')
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.notebook), indent=2))
