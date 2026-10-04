"""Sample image-derived motion at immutable native detections; no GT loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

DETECTOR = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def main(args):
    import numpy as np
    import torch
    import tracksdata as td
    import zarr
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    from backward_flow_model import BackwardFlowNet
    from backward_flow_ops import sample_backward_flow
    from feature_tta_reference import load_reference
    from run_selection import tree_hash
    from tracking_cellmot.io import open_dataset
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('Flow checkpoint checksum mismatch')
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    fold = json.loads(args.manifest.read_text())['folds'][0]
    split_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    diagnostic = fold['train'][::5]
    fitting = [s for s in fold['train'] if s not in diagnostic]
    identity = state['identity']
    if (state['step'] != 1000 or identity['max_steps'] != 1000
        or identity['fitting_stems'] != fitting or identity['diagnostic_stems'] != diagnostic
        or identity['split_sha256'] != split_sha or identity['downsample'] != [1,4,4]
        or any(identity[k] is not False for k in ('public_checkpoint_loaded','selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Exact completed source-only flow model required')
    updates = [int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    if not updates or min(updates) != 1000 or max(updates) != 1000:
        raise ValueError('Incomplete flow optimizer updates')
    reference = load_reference(args.reference,DETECTOR,split_sha,fold['selection'])
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Two T4 devices required')
    torch.set_num_threads(2)
    core = BackwardFlowNet().cuda()
    core.load_state_dict(state['model'],strict=True)
    model = torch.nn.DataParallel(core).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    records = []
    started = time.monotonic()
    for stem in fold['selection']:
        path = args.reference/'outputs'/(stem+'.geff')
        if tree_hash(path) != reference['records'][stem]['graph_sha256']:
            raise ValueError('Native graph checksum mismatch')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        coords = graph.node_attrs().select(['t','z','y','x']).to_numpy().astype(float)
        ds = open_dataset(args.data/stem,normalize=False,load_image=False,require_tracks=False,downsample=(1,4,4))
        if ds.tracks is not None:
            raise ValueError('No ground truth may be loaded in motion inference')
        array = zarr.open_group(str(ds.zarr_path),mode='r')['0']
        shape = tuple(array.shape)
        if shape != tuple(reference['records'][stem]['image_shape']) or shape[0] != 100:
            raise ValueError('Expected exact complete reference movie shape')
        if (coords.shape != (reference['records'][stem]['predicted_nodes'],4)
            or not np.isfinite(coords).all() or (coords<0).any() or (coords>=shape).any()
            or (coords[:,0] != np.floor(coords[:,0])).any()):
            raise ValueError('Invalid native detections')
        low,high = float(ds.quantiles['0.001']),float(ds.quantiles['0.999'])
        if not np.isfinite([low,high]).all() or high <= low:
            raise ValueError('Invalid quantile normalization')
        flows = np.zeros((len(coords),3),dtype=np.float32)
        covered = coords[:,0] == 0
        pairs = 0
        with torch.no_grad():
            for start in range(0,shape[0]-1,2):
                times = list(range(start,min(start+2,shape[0]-1)))
                batches = []
                for t in times:
                    raw = array[t:t+2,::1,::4,::4].astype(np.float32)
                    batches.append(torch.from_numpy((raw-low)/(high-low+1e-6)).clamp(0).half())
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    output = model(torch.stack(batches).cuda().float())
                for b,t in enumerate(times):
                    indices = np.flatnonzero(coords[:,0] == t+1)
                    points = torch.tensor(coords[indices,1:]/[1,4,4],device='cuda',dtype=torch.float32)[None]
                    sampled,valid = sample_backward_flow(output[b:b+1],points)
                    if not bool(valid.all()):
                        raise ValueError('Native node outside flow grid; no clamping or deletion')
                    flows[indices] = sampled[0].cpu().numpy()
                    covered[indices] = True
                    pairs += 1
        if not covered.all() or pairs != shape[0]-1 or not np.isfinite(flows).all():
            raise ValueError('Incomplete/nonfinite full-movie motion')
        destination = args.output/(stem+'.npz')
        np.savez_compressed(destination,coords=coords,backward_um=flows)
        records.append(dict(stem=stem,image_shape=list(shape),processed_frames=shape[0],processed_pairs=pairs,
            predicted_nodes=len(coords),all_nodes_covered=True,native_graph_sha256=reference['records'][stem]['graph_sha256'],
            file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        print(json.dumps(records[-1]),flush=True)
    result = dict(status='completed',run_id='backward-flow-selection-v1',checkpoint_sha256=args.sha256,
        detector_checkpoint_sha256=DETECTOR,split_sha256=split_sha,records=records,
        reference_manifest_sha256=reference['manifest_sha256'],elapsed_seconds=time.monotonic()-started,
        ground_truth_opened=False,target_audit_opened=False,authorized_for_submission=False)
    (args.output/'flow_manifest.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    main(parser.parse_args())
