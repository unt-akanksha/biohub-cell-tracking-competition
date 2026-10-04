"""Cache frozen owned features on FOCUS nodes, with previous GPU replay first."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    sources=json.loads((args.runtime/'source_hashes.json').read_text())
    if any(sha(args.runtime/name)!=digest for name,digest in sources.items()):
        raise ValueError('Exact staged feature-extraction runtime required')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import torch
    import zarr
    from train_unet_transformer import UNetNodeTransformer,TemporalUNet3D,extract_pos_features
    from independent_real_baseline import install_empty_attention_guard
    from image_motion_residual import embedded_flow,flow_hash
    from raw_centroid_flow_sampling import sample_raw_centroid_flow
    from focus_cached_pair import validate_pair
    from focus_feature_spec import validate_spec
    from focus_adaptation_cache_contract import scope
    from tracking_cellmot.io import open_dataset
    started=time.monotonic();torch.set_num_threads(2)
    spec=json.loads((args.runtime/'features_spec.json').read_text())
    validate_spec(spec,scope(args.manifest.read_bytes()))
    if sha(args.checkpoint)!=CHECKPOINT_SHA or sha(args.manifest)!=spec['contract']['split_sha256']:
        raise ValueError('Exact owned checkpoint and original split required')
    if torch.cuda.device_count()!=2:raise ValueError('Exactly two CUDA devices required')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    split=json.loads(args.manifest.read_text())['folds'][0]
    if state['step']!=1000 or state['identity']['training_stems']!=split['train']:raise ValueError('Frozen completed training identity changed')
    model=UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda(0)
    model.load_state_dict(state['model'],strict=True);install_empty_attention_guard(model);model.requires_grad_(False).eval()
    flow=embedded_flow(state,'cuda:1');before=dict(neural=flow_hash(model),flow=flow_hash(flow))
    if before['neural']!=spec['probe_model_tensor_sha256']:raise ValueError('Exact successful neural-probe tensors required')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'raw_detections').mkdir()
    (args.output/'features_spec.json').write_text(json.dumps(spec,indent=2))
    records=[];replayed=[]
    for item in spec['movies']:
        stem=item['stem'];is_replay=item['role']=='replay'
        if not is_replay and replayed!=spec['contract']['replay_stems']:raise ValueError('Both prior GPU replays must pass before new cache')
        raw=args.raw_root/'raw_detections'/(stem+'.npz')
        if sha(raw)!=item['raw_sha256']:raise ValueError('Raw FOCUS cache identity changed')
        with np.load(raw,allow_pickle=False) as data:
            coords=data['coords'].copy();shape=data['movie_shape'].copy()
        frames=3 if is_replay else 100
        if list(shape)!=[frames,64,256,256] or max(int((coords[:,0]==t).sum()) for t in range(frames))>2048:
            raise ValueError('Complete bounded movie required; no truncation')
        if stem not in split['train'] or stem in split['selection']+split['audit_order']:raise ValueError('Original training images only')
        shutil.copyfile(raw,args.output/'raw_detections'/raw.name)
        ds=open_dataset(args.data/stem,normalize=False,load_image=False,require_tracks=False,downsample=(1,4,4))
        if ds.tracks is not None:raise ValueError('No raw annotation graph IO on GPU')
        array=zarr.open_group(str(ds.zarr_path),mode='r')['0']
        if tuple(array.shape)!=(100,64,256,256):raise ValueError('Native image shape required')
        low,high=float(ds.quantiles['0.001']),float(ds.quantiles['0.999'])
        if not np.isfinite([low,high]).all() or high<=low:raise ValueError('Finite original normalization required')
        destination=args.output/stem;destination.mkdir();pair_records=[]
        reference=None
        if is_replay:
            ref=args.probe_root/'outputs'/(stem+'.npz')
            if sha(ref)!=item['probe_sha256']:raise ValueError('Successful previous neural probe required')
            with np.load(ref,allow_pickle=False) as data:reference={k:data[k].copy() for k in data.files}
            if not np.array_equal(coords,reference['coords']):raise ValueError('Replay raw nodes changed')
        window_labels={row['source_frame']:row for row in item.get('windows',[])}
        with torch.no_grad():
            for t in range(frames-1):
                ids=[np.flatnonzero(coords[:,0]==f) for f in (t,t+1)]
                points=[coords[index,1:].astype(np.float64) for index in ids]
                image=np.maximum((np.asarray(array[t:t+2,::1,::4,::4],dtype=np.float32)-low)/(high-low+1e-6),0)
                features,_=model.encode(torch.from_numpy(image).unsqueeze(0).cuda(0))
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    field=flow(torch.from_numpy(image).unsqueeze(0).to('cuda:1').half().float())[0].float().cpu().numpy()
                sampled,sampling=sample_raw_centroid_flow(field,points[1],(64,256,256))
                native=[torch.tensor(p,dtype=torch.float32,device='cuda:0').unsqueeze(0) for p in points]
                grid=[p/p.new_tensor([1,4,4]) for p in native]
                masks=[torch.ones((1,len(p)),dtype=torch.bool,device='cuda:0') for p in points]
                pos=[]
                for i,p in enumerate(points):
                    relative=np.column_stack([np.full(len(p),i),p/np.array([1,4,4])])
                    pos.append(extract_pos_features(relative,(2,64,64,64)))
                indexed=[model._index_features(features[:,i],grid[i],masks[i]) for i in range(2)]
                position=[torch.from_numpy(p).unsqueeze(0).cuda(0) for p in pos]
                logits=(model.predict_edges(*indexed,*native,*position,*masks)[0].float().cpu().numpy()
                        if all(len(p) for p in points) else np.empty((len(points[0]),len(points[1])),np.float32))
                if not np.isfinite(logits).all():raise ValueError('Nonfinite frozen neural features/logits')
                labels=np.full(len(points[1]),-1,np.int64)
                if not is_replay:
                    row=window_labels[t]
                    if row['target_count']!=len(labels) or row['source_count']!=len(points[0]):raise ValueError('Audited labels changed raw frame counts')
                    labels[np.asarray(row['columns'],dtype=np.int64)]=np.asarray(row['parent_rows'],dtype=np.int64)
                packet=dict(source_frame=np.array(t,np.int64),source_indices=ids[0].astype(np.int64),target_indices=ids[1].astype(np.int64),
                    source_coords=coords[ids[0],1:],target_coords=coords[ids[1],1:],
                    source_features=indexed[0][0].float().cpu().numpy(),target_features=indexed[1][0].float().cpu().numpy(),
                    source_pos=pos[0],target_pos=pos[1],backward_um=sampled,labels=labels)
                validated=validate_pair(packet,coords)
                path=destination/f'{t:03d}.npz';np.savez_compressed(path,**packet)
                with np.load(path,allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[k],v) for k,v in packet.items()):raise ValueError('Feature cache round trip changed arrays')
                    restored=[torch.from_numpy(saved[k]).unsqueeze(0).cuda(0) for k in
                        ('source_features','target_features','source_coords','target_coords','source_pos','target_pos')]
                    replay=(model.predict_edges(*restored,*masks)[0].float().cpu().numpy()
                            if all(len(p) for p in points) else logits)
                    if not np.array_equal(logits,replay):raise ValueError('Cached-feature head replay changed logits')
                if is_replay:
                    if not np.array_equal(logits,reference[f'neural_{t}']):raise ValueError('Original GPU neural matrix replay failed')
                    expected_flow=reference['backward_um'][ids[1]]
                    if not np.allclose(sampled,expected_flow,rtol=0,atol=1e-5):raise ValueError('Frozen physical-flow replay exceeds1e-5um')
                    image_sha=hashlib.sha256(image.tobytes()).hexdigest()
                    if image_sha!=item['normalized_image_sha256'][t]:raise ValueError('Original normalized training pixels changed')
                pair_records.append(dict(file=path.name,sha256=sha(path),**validated,sampling=sampling))
        if is_replay:replayed.append(stem)
        record=dict(stem=stem,role=item['role'],raw_sha256=item['raw_sha256'],pairs=pair_records,frames=frames,nodes=len(coords))
        (destination/'manifest.json').write_text(json.dumps(record,indent=2));records.append(record)
        (args.output/'progress.json').write_text(json.dumps(dict(completed_stems=[r['stem'] for r in records],elapsed_seconds=time.monotonic()-started),indent=2))
        print(json.dumps(dict(stem=stem,role=item['role'],cached_pairs=len(pair_records),elapsed_seconds=time.monotonic()-started)),flush=True)
    after=dict(neural=flow_hash(model),flow=flow_hash(flow))
    if before!=after:raise ValueError('Frozen encoder/head/flow weights changed')
    result=dict(status='completed_focus_adaptation_feature_cache',records=records,before=before,after=after,
        replayed_stems=replayed,checkpoint_sha256=CHECKPOINT_SHA,features_spec_sha256=sha(args.runtime/'features_spec.json'),
        labels_from_verified_training_cache=True,raw_gt_graphs_opened=False,source_selection_opened=False,new_target_movies_opened=0,
        optimizer_run=False,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    result['runtime_source_hashes']=sources
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','data','output','checkpoint','raw-root','probe-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
