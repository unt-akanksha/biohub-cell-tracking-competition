"""Six training frames; cache owned neural logits before any GT evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def main(args):
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA or hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=SPLIT_SHA:
        raise ValueError('Exact owned trained checkpoint and split required')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import torch
    import zarr
    from train_unet_transformer import UNetNodeTransformer,TemporalUNet3D,extract_pos_features
    from independent_real_baseline import install_empty_attention_guard
    from tracking_cellmot.io import open_dataset
    from focus_owned_neural_links import link
    started=time.monotonic();torch.set_num_threads(2)
    if torch.cuda.device_count()!=2: raise ValueError('Two visible T4 GPUs required')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    split=json.loads(args.manifest.read_text());fold=split['folds'][0]
    if state['step']!=1000 or state['identity']['training_stems']!=fold['train'] or not state['identity'].get('image_motion'):
        raise ValueError('Complete original training profile required')
    inputs=json.loads((args.runtime/'training_inputs.json').read_text())
    if [r['stem'] for r in inputs]!=['6bba_f1fde7e0','6bba_23af9eeb'] or not all(r['stem'] in fold['train'] for r in inputs):
        raise ValueError('Frozen two training stems required')
    model=UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda()
    model.load_state_dict(state['model'],strict=True);install_empty_attention_guard(model);model.requires_grad_(False).eval()
    def tensors():
        digest=hashlib.sha256()
        for name,value in model.state_dict().items():
            digest.update(name.encode()+b'\0');digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before=tensors();args.output.mkdir(parents=True,exist_ok=False);records=[]
    with torch.no_grad():
        for row in inputs:
            stem=row['stem'];coords=np.asarray(row['coords'],dtype=np.float64);flow=np.asarray(row['backward_um'],dtype=np.float32)
            if coords.shape[1]!=4 or (coords[:,0]>2).any() or set(coords[:,0])!={0.,1.,2.}:raise ValueError('First three raw frames required')
            ds=open_dataset(args.data/stem,normalize=False,load_image=False,require_tracks=False,downsample=(1,4,4))
            if ds.tracks is not None:raise ValueError('No GT access')
            array=zarr.open_group(str(ds.zarr_path),mode='r')['0']
            if tuple(array.shape)!=(100,64,256,256):raise ValueError('Native image shape required')
            low,high=float(ds.quantiles['0.001']),float(ds.quantiles['0.999'])
            if not np.isfinite([low,high]).all() or high<=low:raise ValueError('Finite normalization required')
            matrices={}; images_sha=[]
            for t in range(2):
                image=(np.asarray(array[t:t+2,::1,::4,::4],dtype=np.float32)-low)/(high-low+1e-6)
                image=np.maximum(image,0);images_sha.append(hashlib.sha256(image.tobytes()).hexdigest())
                features,_=model.encode(torch.from_numpy(image).unsqueeze(0).cuda())
                points=[coords[coords[:,0]==f] for f in (t,t+1)]
                if min(map(len,points))==0 or max(map(len,points))>2048:raise ValueError('Bounded nonempty smoke frames required')
                native=[torch.tensor(p[:,1:],dtype=torch.float32,device='cuda').unsqueeze(0) for p in points]
                grid=[p/p.new_tensor([1,4,4]) for p in native]
                masks=[torch.ones((1,len(p)),dtype=torch.bool,device='cuda') for p in points]
                pos=[]
                for index,p in enumerate(points):
                    relative=p.copy();relative[:,0]=index;relative[:,1:]/=[1,4,4]
                    pos.append(torch.from_numpy(extract_pos_features(relative,(2,64,64,64))).unsqueeze(0).cuda())
                indexed=[model._index_features(features[:,i],grid[i],masks[i]) for i in range(2)]
                logits=model.predict_edges(*indexed,*native,*pos,*masks)[0].float().cpu().numpy()
                if not np.isfinite(logits).all():raise ValueError('Nonfinite real learned logits')
                # Direct frozen replay of identical pair must be exact in eval mode.
                replay=model.predict_edges(*indexed,*native,*pos,*masks)[0].float().cpu().numpy()
                if not np.array_equal(logits,replay):raise ValueError('Frozen neural-head replay mismatch')
                matrices[t]=logits
            zeros={t:np.zeros_like(v) for t,v in matrices.items()}
            control=link(coords,flow,zeros);candidate=link(coords,flow,matrices)
            path=args.output/(stem+'.npz')
            np.savez_compressed(path,coords=coords,backward_um=flow,neural_0=matrices[0],neural_1=matrices[1],
                control_edges=np.asarray(control,dtype=float).reshape(-1,3),candidate_edges=np.asarray(candidate,dtype=float).reshape(-1,3))
            record=dict(stem=stem,nodes=len(coords),control_edges=len(control),candidate_edges=len(candidate),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_sha256=row['source_sha256'],normalized_image_sha256=images_sha,
                neural_min=min(float(v.min()) for v in matrices.values()),neural_max=max(float(v.max()) for v in matrices.values()))
            records.append(record);print(json.dumps(record),flush=True)
    after=tensors()
    if before!=after:raise ValueError('Frozen model mutated')
    result=dict(status='completed_focus_owned_neural_probe',records=records,checkpoint_sha256=CHECKPOINT_SHA,
        split_sha256=SPLIT_SHA,model_before=before,model_after=after,ground_truth_opened=False,
        authorized_for_submission=False,elapsed_seconds=time.monotonic()-started,
        feature_sampling='Official integer floor with boundary feature-value extension; exact native coordinates retained',
        positional_features='Window-relative times and fractional downsampled coordinates',
        linking_policy=dict(neural_weight=1,physical_weight=1,null_logit=-4.5,posterior_threshold=.5,max_children=2,max_parents=1))
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','data','output','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
