"""Three training frames, paired native/TTA inference, strict node preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main(args):
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('Source checkpoint hash mismatch')
    sys.path[:0] = [str(args.repo/'scripts'), str(args.repo/'src'), str(args.runtime)]
    import torch
    import tracksdata as td
    from real_checkpoint_gpu_smoke import run_smoke
    from edge_feature_tta import install_edge_feature_tta
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA probe required')
    torch.set_num_threads(2)
    split = json.loads(args.manifest.read_text())
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    fold = split['folds'][0]
    if (state['step'] != 1000 or state['identity']['training_stems'] != fold['train']
        or not state['identity'].get('joint_training') or state['identity'].get('bn_recalibration')):
        raise ValueError('Original full-source joint checkpoint required')
    stem = fold['train'][0]
    if stem in fold['selection'] + fold['audit_order']:
        raise ValueError('Probe cannot open evaluation labels')
    args.output.mkdir(parents=True,exist_ok=True)
    control = run_smoke(args.checkpoint,args.data/stem,args.output/'control')
    candidate = run_smoke(args.checkpoint,args.data/stem,args.output/'candidate',encode_patch=install_edge_feature_tta)
    def coordinates(arm):
        graph = td.graph.IndexedRXGraph.from_geff(str(args.output/arm/'gpu_smoke.geff'))[0]
        return sorted(tuple(row[k] for k in ('t','z','y','x')) for row in graph.node_attrs().iter_rows(named=True))
    native, augmented = coordinates('control'), coordinates('candidate')
    if not native or native != augmented:
        raise ValueError('Feature-only augmentation changed detector nodes')
    receipt = candidate['encode_patch_receipt']
    if receipt['views'] != 8 or receipt['encode_calls'] < 1 or receipt['maximum_mean_absolute_feature_delta'] <= 0:
        raise ValueError('Feature averaging did not run or was a no-op')
    result = dict(status='passed',checkpoint_sha256=args.sha256,movie=stem,frames=3,
        nodes_identical=True,node_coordinates_sha256=hashlib.sha256(json.dumps(native).encode()).hexdigest(),
        control=control,candidate=candidate,selection_opened=False,target_audit_opened=False,
        scope='Training functionality only; no selection or submission authorization',authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    main(parser.parse_args())
