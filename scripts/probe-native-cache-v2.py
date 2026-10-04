"""GPU cache equivalence and throughput on a fixed optimization transition only."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_correspondence_models_v2 import VisualCorrespondence,prepare_patches
from research.native_correspondence_inference_v2 import score_embeddings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();helper=runpy.run_path(str(ROOT/'scripts/train-native-correspondence-v2.py'))
    if helper['pids']():raise RuntimeError('GPU occupied; do not compete with training/diagnostic')
    if args.output.exists():raise ValueError('Preserve prior probe')
    trained=json.loads((args.training/'RESULT.json').read_text())
    data=json.loads((args.data/'RESULT.json').read_text())
    if trained['status']!='training_complete' or trained['data_result_sha256']!=helper['sha'](args.data/'RESULT.json'):
        raise ValueError('Complete matching training/data required')
    records=[r for r in data['records'] if r['embryo']=='6bba' and r['role']=='optimization']
    record=sorted(records,key=lambda r:(r['stem'],r['transition']))[0]
    path=args.data/record['path']
    if helper['sha'](path)!=record['sha256']:raise ValueError('Optimization packet changed')
    with np.load(path,allow_pickle=False) as packet:
        stored=torch.from_numpy(packet['patches']).cuda();ids=torch.from_numpy(packet['ids']).cuda()
        coords=torch.from_numpy(packet['coords']).cuda();valid=torch.from_numpy(packet['valid']).cuda()
    # Never inspect labels or masks: this is only an inference implementation test.
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    results=[];begin=time.monotonic()
    for summary in [m for m in trained['members'] if m['embryo']=='6bba' and m['source_gate']]:
        weights=args.training/(summary['embryo']+'-'+summary['family'])/'best.pt'
        if helper['sha'](weights)!=summary['weights_sha256']:raise ValueError('Selected model changed')
        model=VisualCorrespondence(summary['family'],input_channels=3).cuda().eval()
        model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True)['state_dict'])
        @torch.inference_mode()
        def ordinary():
            all_scores=[]
            for start in range(0,len(ids),12):
                x,c=prepare_patches(stored[ids[start:start+12]],coords[start:start+12],valid[start:start+12])
                with torch.autocast('cuda',dtype=torch.float16):all_scores.append(model(x,c,valid[start:start+12]))
            return torch.cat(all_scores)
        @torch.inference_mode()
        def cached():
            all_embeddings=[]
            for start in range(0,len(stored),64):
                n=min(64,len(stored)-start)
                x,_=prepare_patches(stored[start:start+n,None],torch.zeros((n,1,3),device='cuda'),torch.ones((n,1),dtype=torch.bool,device='cuda'))
                with torch.autocast('cuda',dtype=torch.float16):all_embeddings.append(model.encoder(x[:,0]))
            embeddings=torch.cat(all_embeddings);all_scores=[]
            for start in range(0,len(ids),12):
                with torch.autocast('cuda',dtype=torch.float16):
                    all_scores.append(score_embeddings(model,embeddings[ids[start:start+12]],coords[start:start+12],valid[start:start+12]))
            return torch.cat(all_scores)
        ordinary();cached();torch.cuda.synchronize()
        start=time.monotonic();reference=ordinary();torch.cuda.synchronize();ordinary_time=time.monotonic()-start
        start=time.monotonic();actual=cached();torch.cuda.synchronize();cached_time=time.monotonic()-start
        active=torch.cat((valid[:,1:],valid[:,:1]),dim=-1)
        error=float((reference[active]-actual[active]).abs().max())
        same=torch.equal(reference.argmax(-1),actual.argmax(-1))
        if not same or error>.02:raise ValueError('Cached GPU encoder partition changes choices or logits excessively')
        results.append(dict(family=summary['family'],max_logit_error=error,choices_equal=same,ordinary_seconds=ordinary_time,
                            cached_seconds=cached_time,groups=len(ids),unique_patches=len(stored),weights_sha256=summary['weights_sha256']))
        del model;torch.cuda.empty_cache()
    if not results:raise ValueError('No source-admitted model to test')
    result=dict(status='cache_probe_passed',optimization_packet_sha256=record['sha256'],stem=record['stem'],transition=record['transition'],
                results=results,elapsed_seconds=time.monotonic()-begin,target_labels_opened=False,complete_movie_runtime_tested=False,
                script_sha256=helper['sha'](Path(__file__)))
    helper['save'](args.output,result);print(json.dumps(result))


if __name__=='__main__':main()
