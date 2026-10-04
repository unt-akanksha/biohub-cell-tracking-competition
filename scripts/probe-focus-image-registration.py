"""CPU-only image registration on six frozen training frames; no annotation IO."""
import argparse
import hashlib
import json
from pathlib import Path
import time

STEMS=('6bba_f1fde7e0','6bba_23af9eeb')
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def main(args):
    import numpy as np
    import zarr
    from transition_context import estimate_transition_context
    started=time.monotonic()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=SPLIT_SHA:
        raise ValueError('Frozen split required')
    fold=json.loads(args.manifest.read_text())['folds'][0]
    if not set(STEMS)<=set(fold['train']):
        raise ValueError('Original training images only')
    args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20260910)
    synthetic=rng.normal(size=(25,31,35)).astype(np.float32)
    moved=np.roll(synthetic,(2,-3,4),axis=(0,1,2))
    check=estimate_transition_context(synthetic,moved,voxel_size_zyx_um=(1.625,.40625,.40625))
    if check.global_shift_zyx_voxel!=(2.,-3.,4.):
        raise ValueError('Remote physical/sign functionality failed')
    records=[]
    for stem in STEMS:
        array=zarr.open_group(str(args.data/(stem+'.zarr')),mode='r')['0']
        if tuple(array.shape)!=(100,64,256,256):
            raise ValueError('Native full movie shape required')
        frames=[np.asarray(array[t],dtype=np.float32) for t in range(3)]
        image_hashes=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
        transitions=[]
        for t in range(2):
            forward=estimate_transition_context(frames[t],frames[t+1],voxel_size_zyx_um=(1.625,.40625,.40625))
            reverse=estimate_transition_context(frames[t+1],frames[t],voxel_size_zyx_um=(1.625,.40625,.40625))
            transitions.append(dict(source_frame=t,target_frame=t+1,forward=forward.to_dict(),
                reverse=reverse.to_dict(),inverse_consistency_um=float(np.linalg.norm(
                    np.array(forward.global_shift_zyx_um)+np.array(reverse.global_shift_zyx_um)))))
        row=dict(stem=stem,frame_hashes=image_hashes,transitions=transitions)
        records.append(row)
        (args.output/(stem+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False))
        print(json.dumps(row),flush=True)
    result=dict(status='completed_training_image_registration_probe',records=records,
        split_sha256=SPLIT_SHA,ground_truth_opened=False,gpu_used=False,elapsed_seconds=time.monotonic()-started,
        synthetic_sign_check_passed=True,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.runtime.iterdir() if p.is_file()},authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','data','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int)
    main(parser.parse_args())
