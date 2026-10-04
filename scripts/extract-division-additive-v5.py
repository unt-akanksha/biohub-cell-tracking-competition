"""Small image-only-center extraction; never modifies the old training dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import normalize_gpu,proposals,patches_gpu


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-proof',type=Path);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'));helper['preflight']()
    contract_path=ROOT/'CONTRACT.json';contract=json.loads(contract_path.read_text())
    for r in contract['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Frozen extraction changed')
    recipes=json.loads((ROOT/'plan/RECIPES.json').read_text())['recipes']
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='additive_smoke_passed' or proof['contract_sha256']!=sha(contract_path):raise ValueError('Exact smoke required')
    else:recipes=[next(r for r in recipes if r['embryo']==e) for e in ('44b6','6bba')]
    if args.output.exists():raise ValueError('Do not overwrite existing result')
    args.output.mkdir();(args.output/'data').mkdir()
    archive_plan=json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    if archive_plan['movie_plan_sha256']!='30c914750af6ce3d03af2e39cf9a172cd6dd95817dbd1bfe266b275468346281':raise ValueError('Wrong archive scope')
    archive=helper['Archive'](archive_plan);records=[];totals={}
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    started=time.monotonic();timer=threading.Timer(300,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        for r in recipes:
            helper['preflight']();t=r['transition'];images,_=archive.movie(r,[t,t+1])
            coords=np.array(r['coords'],np.float32);parts=[]
            for frame_time,chosen,indices in ((t,coords[:1],r['proposal_indices'][:1]),(t+1,coords[1:],r['proposal_indices'][1:])):
                if hashlib.sha256(images[frame_time].tobytes()).hexdigest()!=r['raw_sha256'][str(frame_time)]:raise ValueError('Raw source frame changed')
                normalized,pooled=normalize_gpu(images[frame_time],torch);points=proposals(pooled)
                np.testing.assert_array_equal(points[indices],chosen)
                parts.append(patches_gpu(normalized,chosen,torch))
            path=args.output/'data'/f'{r["stem"]}-{t:03d}.npz'
            if path.exists():raise ValueError('Duplicate transition')
            np.savez_compressed(path,patches=np.concatenate(parts),triples=np.array([[0,1,2]],np.int64),labels=np.array([1],np.int64),
                                coords=coords[None],truth_ids=np.array([[r['parent'],*r['children']]],np.int64),
                                daughter_parents=np.array([[r['parent'],r['parent']]],np.int64))
            record=dict(path=path.relative_to(args.output).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                        stem=r['stem'],embryo=r['embryo'],role='optimization',transition=t,patches=3,triples=1,positive=1,negative=0)
            records.append(record);totals[r['embryo']]=totals.get(r['embryo'],0)+1
            print(json.dumps(dict(stem=r['stem'],transition=t,completed=len(records))),flush=True)
        old=json.loads((ROOT/'plan/OLD_DATA.json').read_text())
        result=dict(status='additive_smoke_passed' if args.smoke else 'additive_extraction_complete',records=records,
                    totals=totals,encoders=old['encoders'],contract_sha256=sha(contract_path),seconds=time.monotonic()-started,
                    old_data_sha256=sha(ROOT/'plan/OLD_DATA.json'),selection_opened=False,target_pilot_labels_used=False,
                    authorized_for_submission=False)
        helper['save'](args.output/'RESULT.json',result);helper['save'](args.output/'DATA.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','encoders')}),flush=True)
    finally:timer.cancel()


if __name__=='__main__':main()
