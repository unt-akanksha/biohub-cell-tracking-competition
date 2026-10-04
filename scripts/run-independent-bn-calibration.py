"""Training-image-only BatchNorm calibration, preceded by a two-movie GPU gate."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def calibration_scope(identity,split):
    fold = split['folds'][0]
    stems = identity['training_stems']
    if identity.get('joint_training') is not True or stems != fold['train'] or len(stems) != 120:
        raise ValueError('Completed 120-movie joint training scope required')
    if set(stems) & set(fold['selection']+fold['audit_order']):
        raise ValueError('Calibration may not use evaluation movies')
    return stems


def main(args):
    import numpy as np
    import torch
    import zarr
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    from train_unet_transformer import UNetNodeTransformer,TemporalUNet3D
    from tracking_cellmot.io import open_dataset
    from bn_recalibration import recalibrate_batchnorm
    from real_checkpoint_gpu_smoke import run_smoke
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for calibration')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('Source checkpoint checksum mismatch')
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['step'] != 1000 or state['identity']['max_steps'] != 1000:
        raise ValueError('Completed joint checkpoint required')
    split = json.loads(args.manifest.read_text())
    stems = calibration_scope(state['identity'],split)
    model = UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),
                               unet_out_channels=32,pos_feat_dim=32).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False)
    args.output.mkdir(parents=True,exist_ok=True)
    def batches(movies,records):
        for stem in movies:
            # Explicitly no tracks loaded, and no detection/edge predictions
            # or organizer estimated counts participate in calibration.
            ds = open_dataset(args.data/stem,normalize=False,require_tracks=False,
                              load_image=False,downsample=(1,4,4))
            array = zarr.open_group(str(ds.zarr_path),mode='r')['0']
            if array.shape[0] < 4:
                raise ValueError('Movie too short for fixed calibration windows')
            starts = [(array.shape[0]-2)//4,3*(array.shape[0]-2)//4]
            values = []
            for t in starts:
                raw = array[t:t+2,::1,::4,::4].astype(np.float32)
                values.append(torch.from_numpy((raw-float(ds.quantiles['0.001']))/
                    (float(ds.quantiles['0.999'])-float(ds.quantiles['0.001'])+1e-6)).clamp_min(0).half())
            batch = torch.stack(values)
            if list(batch.shape[2:]) != list(ds.image_shape[1:]):
                raise ValueError('Calibration/preprocessing shape mismatch')
            records.append(dict(stem=stem,window_starts=starts,
                input_sha256=hashlib.sha256(batch.numpy().tobytes()).hexdigest()))
            yield batch.cuda()
    def encoder(target):
        def encode(batch):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                target.encode(batch)
        return encode
    def save(target,path,receipt,phase):
        payload = dict(state)
        payload['model'] = {k:v.detach().cpu() for k,v in target.state_dict().items()}
        payload['identity'] = copy.deepcopy(state['identity'])
        payload['identity']['bn_recalibration'] = dict(status=phase,source_sha256=args.sha256,
            movie_count=receipt['batches'],windows_per_movie=2,augmentation=False,
            calibration_labels_opened=False,**receipt)
        torch.save(payload,path.with_suffix('.tmp.pt'))
        os.replace(path.with_suffix('.tmp.pt'),path)
    # Do not calibrate the full model until the exact implementation has
    # updated a disposable copy and passed serialized real GPU inference.
    probe = copy.deepcopy(model)
    probe_records = []
    probe_receipt = recalibrate_batchnorm(probe,batches(stems[:2],probe_records),encoder(probe))
    probe_path = args.output/'probe.pt'
    save(probe,probe_path,probe_receipt,'probe')
    del probe
    probe_smoke = run_smoke(probe_path,args.data/stems[0],args.output/'probe_smoke')
    if probe_smoke['status'] != 'passed':
        raise RuntimeError('Small GPU calibration gate failed')
    print(json.dumps(dict(event='calibration_probe',**probe_receipt)),flush=True)
    records = []
    receipt = recalibrate_batchnorm(model,batches(stems,records),encoder(model))
    if receipt['batches'] != 120 or [r['stem'] for r in records] != stems:
        raise ValueError('Incomplete frozen training calibration coverage')
    final = args.output/'last.pt'
    save(model,final,receipt,'completed')
    after = run_smoke(final,args.data/stems[0],args.output/'after')
    result = dict(status='completed',source_sha256=args.sha256,receipt=receipt,
        input_records=records,probe_smoke=probe_smoke,after=after,
        checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        calibration_labels_opened=False,optimizer_steps=0,authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='calibration_complete',checkpoint_sha256=result['checkpoint_sha256'],
        **receipt)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    main(parser.parse_args())
