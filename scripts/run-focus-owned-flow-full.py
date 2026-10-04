"""Sample frozen native flow and persist both full graph arms before GT scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def tree_hash(root):
    digest=hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode()+b'\0')
            digest.update(path.read_bytes()+b'\0')
    return digest.hexdigest()


def main(args):
    import numpy as np
    import torch
    import zarr
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    from focus_owned_flow_full_contract import receipt
    from verify_raw_cache import verify
    from probe_runtime import tensor_hash,persist_graph
    from backward_flow_model import BackwardFlowNet
    from raw_centroid_flow_sampling import sample_raw_centroid_flow
    from independent_motion_prior import link_motion
    from backward_flow_linking import link_backward_flow
    from tracking_cellmot.io import open_dataset

    policy=receipt((args.runtime/'verified_probe.json').read_bytes(),args.manifest.read_bytes())
    raw=verify(args.reference)
    if raw['terminal_sha256']!=policy['raw_terminal_sha256'] or [r['stem'] for r in raw['movies']]!=policy['stems']:
        raise ValueError('Frozen complete diagnostic raw cache required')
    if args.sha256!=policy['flow_checkpoint_sha256'] or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('Exact frozen trained flow checkpoint required')
    probe_path=args.probe/'sampled_flow.npz'
    if hashlib.sha256(probe_path.read_bytes()).hexdigest()!=policy['probe_motion_sha256']:
        raise ValueError('Original GPU smoke motion required for exact replay')
    with np.load(probe_path,allow_pickle=False) as saved:
        if set(saved.files)!={'coords','backward_um'}: raise ValueError('Original probe schema required')
        probe_coords=saved['coords'].copy(); probe_flow=saved['backward_um'].copy()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Two-device CUDA execution required')
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    core=BackwardFlowNet().cuda().requires_grad_(False).eval()
    core.load_state_dict(state['model'],strict=True)
    before=tensor_hash(core)
    if before!=policy['flow_tensor_sha256']: raise ValueError('Frozen flow tensors changed')
    model=torch.nn.DataParallel(core).eval()
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]; started=time.monotonic(); probe_replayed=False
    for record in raw['movies']:
        stem=record['stem']
        with np.load(args.reference/'raw_detections'/(stem+'.npz'),allow_pickle=False) as data:
            coords=data['coords'].copy(); shape=tuple(data['movie_shape'].tolist())
        if shape!=(100,64,256,256): raise ValueError('Exact100-frame diagnostic movie required')
        original=coords.copy()
        if max(record['frame_counts'])>2048: raise ValueError('Node guard exceeded; no truncation')
        ds=open_dataset(args.data/stem,normalize=False,load_image=False,require_tracks=False,downsample=(1,4,4))
        if ds.tracks is not None: raise ValueError('No GT before graph freezing')
        array=zarr.open_group(str(ds.zarr_path),mode='r')['0']
        if tuple(array.shape)!=shape: raise ValueError('Raw detections/image mismatch')
        low,high=float(ds.quantiles['0.001']),float(ds.quantiles['0.999'])
        if not np.isfinite([low,high]).all() or high<=low: raise ValueError('Invalid normalization')
        flows=np.zeros((len(coords),3),dtype=np.float32)
        covered=coords[:,0]==0; sampling=[]; pairs=0
        with torch.no_grad():
            for start in range(0,99,2):
                times=list(range(start,min(start+2,99)))
                batches=[torch.from_numpy((array[t:t+2,::1,::4,::4].astype(np.float32)-low)
                    /(high-low+1e-6)).clamp(0).half() for t in times]
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    fields=model(torch.stack(batches).cuda().float())
                for b,t in enumerate(times):
                    indices=np.flatnonzero(coords[:,0]==t+1)
                    field=fields[b].float().cpu().numpy()
                    flows[indices],sample=sample_raw_centroid_flow(field,coords[indices,1:],shape[1:])
                    covered[indices]=True; pairs+=1
                    sampling.append(dict(frame=t+1,**sample))
        if (pairs!=99 or not covered.all() or not np.array_equal(coords,original)
            or not np.isfinite(flows).all()):
            raise ValueError('Complete immutable-coordinate finite motion required')
        replay=None
        if stem==policy['probe_stem']:
            selected=coords[:,0]<3
            if not np.array_equal(coords[selected],probe_coords) or not np.array_equal(flows[selected],probe_flow):
                raise ValueError('Full run failed exact first-three-frame GPU probe replay')
            replay=True; probe_replayed=True
        destination=args.output/stem; destination.mkdir()
        arms={}
        for arm,edges in (('control',link_motion(coords)),('candidate',link_backward_flow(coords,flows))):
            path=destination/(arm+'.geff')
            arms[arm]=dict(**persist_graph(coords,edges,path),graph_sha256=tree_hash(path))
        sampled=destination/'sampled_flow.npz'; np.savez_compressed(sampled,coords=coords,backward_um=flows)
        records.append(dict(stem=stem,image_shape=list(shape),processed_frames=100,processed_pairs=pairs,
            raw_checkpoint_sha256=record['sha256'],coordinate_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
            sample_sha256=hashlib.sha256(sampled.read_bytes()).hexdigest(),graphs=arms,
            all_nodes_covered=True,sampler_receipts=sampling,probe_replayed=replay))
        print(json.dumps({k:v for k,v in records[-1].items() if k!='sampler_receipts'}),flush=True)
    if not probe_replayed or tensor_hash(core)!=before:
        raise ValueError('Frozen tensor identity and actual probe replay required')
    result=dict(status='completed',run_id='focus-owned-flow-full-v1',contract=policy,records=records,
        elapsed_seconds=time.monotonic()-started,frozen_flow_before=before,frozen_flow_after=tensor_hash(core),
        probe_replayed=True,ground_truth_opened=False,target_embryo_images_read=True,
        new_target_movies_opened=0,authorized_for_submission=False)
    (args.output/'flow_manifest.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output','reference','probe'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    main(parser.parse_args())
