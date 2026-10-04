"""Paired detector-only D4 experiment on three training frames, fixed flow."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def main(args):
    if (hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA
        or hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=SPLIT_SHA):
        raise ValueError('Exact independently trained checkpoint and split required')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import torch
    from real_checkpoint_gpu_smoke import run_smoke
    from detector_spatial_tta import install_detector_spatial_tta
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA functionality probe required')
    torch.set_num_threads(2)
    split=json.loads(args.manifest.read_text()); fold=split['folds'][0]
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (state['step']!=1000 or state['identity']['training_stems']!=fold['train']
        or not state['identity'].get('image_motion')):
        raise ValueError('Completed full-source embedded image model required')
    stem=fold['train'][0]
    if stem in fold['selection']+fold['audit_order']:
        raise ValueError('Training-only probe required')
    del state
    args.output.mkdir(parents=True,exist_ok=True)
    control=run_smoke(args.checkpoint,args.data/stem,args.output/'control',standalone_image_flow=True)
    candidate=run_smoke(args.checkpoint,args.data/stem,args.output/'candidate',standalone_image_flow=True,
        pre_motion_patch=install_detector_spatial_tta)
    receipt=candidate['pre_motion_patch_receipt']
    if (receipt['views']!=8 or receipt['encode_calls']<1 or receipt['maximum_mean_absolute_logit_delta']<=0
        or min(control['predicted_nodes'],candidate['predicted_nodes'])<=0):
        raise ValueError('Real detector averaging must execute with finite nonempty detections')
    result=dict(status='passed_detector_tta_functionality',checkpoint_sha256=CHECKPOINT_SHA,split_sha256=SPLIT_SHA,
        movie=stem,frames=3,control=control,candidate=candidate,detector_coordinates_may_change=True,
        linking_policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5),
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
