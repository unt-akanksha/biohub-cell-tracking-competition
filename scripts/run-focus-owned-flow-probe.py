"""Three training frames: exact cached FOCUS centroids plus frozen owned flow.

No detector inference, public linker, labels, target movies, or submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

FLOW_SHA = '3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'
FLOW_TENSOR_SHA = 'e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779'
SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
RAW_SHA = '0609934b1e2a40473763acf521cfcf7120e418f5857c24d6f28c0e662638bd41'
STEM = '6bba_23af9eeb'


def validate_probe_input(root, split_path):
    import numpy as np
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != SPLIT_SHA:
        raise ValueError('Exact frozen split required')
    fold = json.loads(split_path.read_text())['folds'][0]
    if STEM not in fold['train'] or STEM in fold['selection'] + fold['audit_order']:
        raise ValueError('Recorded training-only smoke required')
    terminal_path = root/'focus3d_raw_detections_terminal.json'
    if hashlib.sha256(terminal_path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Exact completed raw-detection terminal required')
    terminal = json.loads(terminal_path.read_text())
    if terminal['status'] != 'completed' or any(terminal[k] is not False for k in (
        'ground_truth_opened','postprocessing_applied','submission_created',
        'competition_test_data_read','public_predictions_copied','metric_hack_used')):
        raise ValueError('Label-blind raw input required')
    rows = [r for r in terminal['raw_detections'] if r['stem'] == STEM]
    if len(rows) != 1:
        raise ValueError('One frozen training movie required')
    record = rows[0]
    path = root/'raw_detections'/(STEM+'.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Raw centroid checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'coords','movie_shape','scale_um'}:
            raise ValueError('Exact raw centroid schema required')
        coords = data['coords'].copy()
        shape = data['movie_shape'].copy()
        if not np.array_equal(data['scale_um'], [1.625,.40625,.40625]):
            raise ValueError('Physical scale mismatch')
    if (shape.tolist() != record['movie_shape'] or shape.tolist() != [100,64,256,256]
        or coords.shape != (record['node_count'],4) or not np.isfinite(coords).all()
        or (coords < 0).any() or (coords > shape-1).any()
        or (coords[:,0] != np.floor(coords[:,0])).any()
        or record['failed_frames'] != 0 or record['postprocessing_applied'] is not False
        or record['coordinate_source'] != 'raw instance centroids'
        or [int((coords[:,0] == t).sum()) for t in range(100)] != record['frame_counts']):
        raise ValueError('Invalid raw centroid coverage/provenance')
    return coords[coords[:,0] < 3].copy(), shape, record, fold


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()+b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def persist_graph(coords, edges, path):
    """Validate topology and exact-node real GEFF serialization before scoring."""
    import numpy as np
    import polars as pl
    import tracksdata as td
    parents,children={},{}
    pairs=set()
    for s,t,p in edges:
        if (int(s)!=s or int(t)!=t or not 0<=s<len(coords) or not 0<=t<len(coords)
            or coords[t,0]!=coords[s,0]+1 or not np.isfinite(p) or not .5<p<=1
            or (s,t) in pairs):
            raise ValueError('Valid distinct next-frame posterior-selected edges required')
        pairs.add((s,t)); parents[t]=parents.get(t,0)+1; children[s]=children.get(s,0)+1
    if any(v>1 for v in parents.values()) or any(v>2 for v in children.values()):
        raise ValueError('One parent and at most two children required')
    graph=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):
        graph.add_node_attr_key(axis,pl.Float64,0.)
    ids=graph.bulk_add_nodes([dict(t=int(c[0]),z=float(c[1]),y=float(c[2]),x=float(c[3])) for c in coords])
    if edges:
        graph.bulk_add_edges([dict(source_id=ids[s],target_id=ids[t]) for s,t,p in edges])
    graph.to_geff(path)
    restored=td.graph.IndexedRXGraph.from_geff(str(path))[0]
    roundtrip=restored.node_attrs().sort('node_id').select('t','z','y','x').to_numpy()
    if not np.array_equal(roundtrip,coords) or restored.num_edges()!=len(edges):
        raise ValueError('Exact-node GEFF round trip failed')
    return dict(nodes=len(coords),edges=len(edges),exact_coordinate_round_trip=True)


def main(args):
    import numpy as np
    import torch
    import zarr
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    from backward_flow_model import BackwardFlowNet
    from backward_flow_ops import sample_backward_flow
    from raw_centroid_flow_sampling import sample_raw_centroid_flow
    from backward_flow_linking import link_backward_flow
    from independent_motion_prior import link_motion
    from tracking_cellmot.io import open_dataset

    started = time.monotonic()
    coords, shape, record, fold = validate_probe_input(args.reference, args.manifest)
    original = coords.copy()
    if args.sha256 != FLOW_SHA or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != FLOW_SHA:
        raise ValueError('Exact completed owned flow checkpoint required')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    identity = state['identity']; diagnostic = fold['train'][::5]
    if (state['step'] != 1000 or identity['max_steps'] != 1000
        or identity['fitting_stems'] != [s for s in fold['train'] if s not in diagnostic]
        or identity['diagnostic_stems'] != diagnostic or identity['split_sha256'] != SPLIT_SHA
        or identity['downsample'] != [1,4,4]
        or any(identity[k] is not False for k in (
            'public_checkpoint_loaded','selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Frozen independent flow training scope mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Two-device CUDA functionality gate required')
    torch.set_num_threads(2)
    core = BackwardFlowNet().cuda().eval().requires_grad_(False)
    core.load_state_dict(state['model'], strict=True)
    before = tensor_hash(core)
    if before != FLOW_TENSOR_SHA:
        raise ValueError('Frozen flow tensor identity mismatch')
    model = torch.nn.DataParallel(core).eval()
    ds = open_dataset(args.data/STEM, normalize=False, load_image=False,
                      require_tracks=False, downsample=(1,4,4))
    if ds.tracks is not None:
        raise ValueError('No labels in functionality probe')
    array = zarr.open_group(str(ds.zarr_path), mode='r')['0']
    if tuple(array.shape) != tuple(shape):
        raise ValueError('Cached detections/image shape mismatch')
    low, high = float(ds.quantiles['0.001']), float(ds.quantiles['0.999'])
    if not np.isfinite([low,high]).all() or high <= low:
        raise ValueError('Invalid movie normalization')
    batches = [torch.from_numpy((array[t:t+2,::1,::4,::4].astype(np.float32)-low)
               /(high-low+1e-6)).clamp(0).half() for t in (0,1)]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        fields = model(torch.stack(batches).cuda().float())
    flows = np.zeros((len(coords),3), dtype=np.float32)
    sampled_rows = []; parity = []
    for batch, t in enumerate((1,2)):
        rows = np.flatnonzero(coords[:,0] == t)
        field = fields[batch].float().cpu().numpy()
        flows[rows], receipt = sample_raw_centroid_flow(field, coords[rows,1:], shape[1:])
        grid = coords[rows,1:] / [1,4,4]
        interior = np.all(grid <= np.asarray(field.shape[1:])-1, axis=1)
        points = torch.tensor(grid[interior], device='cuda', dtype=torch.float32)[None]
        reference, valid = sample_backward_flow(fields[batch:batch+1], points)
        if not bool(valid.all()):
            raise ValueError('Interior reference incorrectly classified')
        difference = float(np.max(np.abs(flows[rows[interior]]-reference[0].cpu().numpy()), initial=0))
        if difference > 2e-5:
            raise ValueError('Interior sampling differs from native Torch flow sampler')
        sampled_rows.append(dict(frame=t, **receipt)); parity.append(difference)
    if not np.array_equal(coords,original) or tensor_hash(core) != before or not np.any(flows != 0):
        raise ValueError('Expected nontrivial frozen motion with immutable raw centroids')
    control = link_motion(coords)
    zero = link_backward_flow(coords, np.zeros_like(flows))
    candidate = link_backward_flow(coords, flows)
    if [(s,t) for s,t,p in control] != [(s,t) for s,t,p in zero]:
        raise ValueError('Zero-flow control does not reproduce static topology')
    # Persist both graph arms and verify the real GEFF round trip, not only arrays.
    args.output.mkdir(parents=True, exist_ok=False)
    graphs = {}
    for arm, edges in (('control',control),('candidate',candidate)):
        graphs[arm] = persist_graph(coords,edges,args.output/(arm+'.geff'))
    cache = args.output/'sampled_flow.npz'
    np.savez_compressed(cache,coords=coords,backward_um=flows)
    result = dict(status='passed_cached_centroid_flow_functionality',movie=STEM,frames=3,
        processed_pairs=2,checkpoint_sha256=FLOW_SHA,frozen_flow_tensor_sha256=before,
        split_sha256=SPLIT_SHA,raw_terminal_sha256=RAW_SHA,raw_checkpoint_sha256=record['sha256'],
        coordinate_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
        sample_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),sampler_receipts=sampled_rows,
        maximum_interior_parity_error_um=max(parity),static_zero_flow_topology_identical=True,
        graphs=graphs,elapsed_seconds=time.monotonic()-started,ground_truth_opened=False,
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False,
        scope='Three recorded training frames; functionality, not accuracy')
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    main(parser.parse_args())
