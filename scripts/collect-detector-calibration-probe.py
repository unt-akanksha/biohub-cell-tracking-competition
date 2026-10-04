"""Only six training movies: frozen FP32 parent/sparse confidence collection.

No fitting, source-selection access, target access or submission generation.
Run only in a separately staged small GPU probe after CPU matching tests pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PARENT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPARSE_SHA='0f441c6e8f1e649bd1549ef519af5ada43d4b01122b96934685d28cbb53b2aa7'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def main(args):
    for path,expected in ((args.parent,PARENT_SHA),(args.candidate,SPARSE_SHA),(args.manifest,SPLIT_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen checkpoint and original split bytes required')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import polars as pl
    import torch
    import zarr
    import predict_unet_transformer as infer
    import train_unet_transformer as official
    from detector_calibration_records import calibration_scope,select_frames,temporal_context
    from detector_spatial_tta import install_detector_spatial_tta
    from image_motion_residual import flow_hash
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise ValueError('Two isolated Kaggle CUDA devices required')
    torch.set_num_threads(2)
    split=json.loads(args.manifest.read_text()); groups=calibration_scope(split,probe=True)
    models={}; before={}
    for index,(name,path) in enumerate((('parent',args.parent),('candidate',args.candidate))):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['identity']['training_stems']!=split['folds'][0]['train'] or state['step']!=1000:
            raise ValueError('Completed original source-training membership required')
        if name=='candidate' and (state['identity'].get('owned_detector_objective')!='sparse'
            or state['identity'].get('fine_tuning_stems')!=split['folds'][0]['train']):
            raise ValueError('Only the completed sparse-control checkpoint is allowed')
        model=official.UNetNodeTransformer(official.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),
            unet_out_channels=32,pos_feat_dim=32).to(f'cuda:{index}').eval().requires_grad_(False)
        model.load_state_dict(state['model'],strict=True)
        before[name]=flow_hash(model); install_detector_spatial_tta(model)
        models[name]=(model,torch.device(f'cuda:{index}'))
        del state
    cutoff=float(torch.sigmoid(torch.tensor(.3,dtype=torch.float32)))
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]; started=time.monotonic()
    for group,stems in groups.items():
        for stem in stems:
            ds=infer.open_dataset(args.data/stem,normalize=False,load_image=False,require_tracks=True,downsample=(1,4,4))
            attrs=ds.tracks.node_attrs(attr_keys=['node_id','t','z','y','x'])
            times=select_frames(stem,attrs['t'].to_list(),ds.image_shape[0])
            arr=zarr.open_group(str(ds.zarr_path),mode='r')['0']
            low=float(ds.quantiles['0.001']); high=float(ds.quantiles['0.999'])
            kernel=infer.pool_kernel_from_um(5.,tuple(s*d for s,d in zip(ds.scale,(1,4,4))))
            for t in times:
                if time.monotonic()-started>2400: raise RuntimeError('Bounded six-movie probe exceeded40minutes')
                context,w=temporal_context(t,ds.image_shape[0])
                images=torch.stack([infer._load_frame(arr,f,list(ds.image_shape[1:]),(1,4,4)) for f in context])
                images=((images-low)/(high-low+1e-6)).clamp(0.).unsqueeze(0)
                if images.dtype!=torch.float32 or not torch.isfinite(images).all():
                    raise ValueError('Exact finite FP32 inference input required')
                annotations=attrs.filter(pl.col('t')==t).sort('node_id')
                payload=dict(truth_coords=annotations.select('t','z','y','x').to_numpy(),truth_ids=annotations['node_id'].to_numpy())
                row=dict(stem=stem,group=group,t=t,context=list(context),annotation_count=len(annotations),
                    input_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest())
                for name,(model,device) in models.items():
                    with torch.no_grad():
                        _,outputs=model.encode(images.to(device))
                        logits=outputs[w][0]
                        if logits.dtype!=torch.float32: raise ValueError('No AMP/FP16 quantization in this probe')
                        peaks=infer._detect_cells_pooled(logits,t,cutoff,kernel)
                        indices=torch.as_tensor(peaks[:,1:].astype(np.int64),device=device)
                        probabilities=logits.sigmoid()[0,indices[:,0],indices[:,1],indices[:,2]].cpu().numpy()
                    if len(peaks)>4096: raise ValueError('Excessive probe peak count; abort without truncation')
                    payload[name+'_coords']=peaks.astype(np.float64)*np.array([1,1,4,4])
                    payload[name+'_probabilities']=probabilities
                    row[name+'_peaks']=len(peaks)
                path=args.output/f'{stem}_{t:03d}.npz'
                np.savez_compressed(path,**payload)
                row.update(artifact=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                records.append(row); print(json.dumps(row),flush=True)
    if len(records)!=18 or any(flow_hash(model)!=before[name] for name,(model,_) in models.items()):
        raise ValueError('Exactly18 frame records and unchanged model tensors required')
    result=dict(status='completed_training_calibration_collection_probe',groups=groups,records=records,
        baseline_threshold=cutoff,parent_sha256=PARENT_SHA,candidate_sha256=SPARSE_SHA,split_sha256=SPLIT_SHA,
        elapsed_seconds=time.monotonic()-started,models_unchanged=True,precision='FP32 exact inference',
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False,
        diagnostic_only=True,calibration_fitted=False,
        caveat='Six neural-training movies only; functionality records, not independent validation or full calibration')
    (args.output/'collection_result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','parent','candidate','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
