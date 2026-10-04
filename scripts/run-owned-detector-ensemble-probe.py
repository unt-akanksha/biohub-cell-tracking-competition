"""Three-training-frame fixed ensemble smoke, no held-out labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PARENT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
PU_SHA='b07f39a930855c1493e43ad16626643d6b666dc8c4dcc1426a4145cdcaff2cdf'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
PROBE_SHA='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba'


def tensor_hash(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    return digest.hexdigest()


def guard(model):
    original=model.predict_edges
    receipt=dict(maximum_nodes=0,limit=2048,truncation=False)
    def predict(*values):
        count=max(values[0].shape[1],values[1].shape[1])
        receipt['maximum_nodes']=max(receipt['maximum_nodes'],count)
        if count>2048:
            raise RuntimeError('Node guard exceeded; no truncation allowed')
        return original(*values)
    model.predict_edges=predict
    return receipt


def main(args):
    for path,expected in ((args.checkpoint,PARENT_SHA),(args.secondary,PU_SHA),
                          (args.manifest,SPLIT_SHA),(args.runtime/'owned_probe.json',PROBE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen checkpoint/split/probe hash mismatch')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import torch
    from real_checkpoint_gpu_smoke import run_smoke
    from detector_spatial_tta import install_detector_spatial_tta
    from owned_detector_ensemble import install_owned_detector_ensemble
    from selection_contract import check_completed_profile,owned_detector_fit_receipt
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA probe required')
    torch.set_num_threads(2)
    split=json.loads(args.manifest.read_text()); fold=split['folds'][0]
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    secondary=torch.load(args.secondary,map_location='cpu',weights_only=True)
    probe=json.loads((args.runtime/'owned_probe.json').read_text())
    check_completed_profile(parent)
    provenance=owned_detector_fit_receipt(secondary,probe)
    for state in (parent,secondary):
        if state['identity']['training_stems']!=fold['train']:
            raise ValueError('Exact training scope required')
    stem=fold['train'][0]
    if stem in fold['selection']+fold['audit_order']:
        raise ValueError('Training-only smoke required')
    kept=[]
    def control_patch(model):
        model.requires_grad_(False)
        kept.append((model,tensor_hash(model)))
        return install_detector_spatial_tta(model)
    def candidate_patch(model):
        other=copy.deepcopy(model)
        other.load_state_dict(secondary['model'],strict=True)
        other.eval()
        for item in (model,other):
            kept.append((item,tensor_hash(item)))
        return install_owned_detector_ensemble(model,other)
    args.output.mkdir(parents=True,exist_ok=True)
    control=run_smoke(args.checkpoint,args.data/stem,args.output/'control',
        standalone_image_flow=True,pre_motion_patch=control_patch,encode_patch=guard)
    candidate=run_smoke(args.checkpoint,args.data/stem,args.output/'candidate',
        standalone_image_flow=True,pre_motion_patch=candidate_patch,encode_patch=guard)
    frozen=[dict(before=before,after=tensor_hash(model)) for model,before in kept]
    receipt=candidate['pre_motion_patch_receipt']
    if (any(row['before']!=row['after'] for row in frozen)
        or receipt['encode_calls']<1 or receipt['maximum_mean_absolute_logit_delta']<=0
        or min(control['predicted_nodes'],candidate['predicted_nodes'])<=0):
        raise ValueError('Nonempty executed frozen ensemble required')
    result=dict(status='passed_owned_detector_ensemble_functionality',
        parent_sha256=PARENT_SHA,secondary_sha256=PU_SHA,split_sha256=SPLIT_SHA,
        movie=stem,frames=3,control=control,candidate=candidate,frozen_hashes=frozen,
        secondary_provenance=provenance,selection_opened=False,target_audit_opened=False,
        authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','secondary','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
