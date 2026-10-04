"""Small native-image division enrichment, only frozen optimization transitions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import normalize_gpu,proposals,patches_gpu
from research.native_division_data_v3 import native_bindings,triplets


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-proof',type=Path);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'));helper['preflight']()
    contract=json.loads((ROOT/'CONTRACT.json').read_text())
    for record in contract['files']:
        if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Enrichment bundle changed')
    movie_path=ROOT/'plan/MOVIES.json';movie_plan=json.loads(movie_path.read_text())
    if sha(movie_path)!='2ec1feed3e86a55e1696dbca4ecbee14c1deebfd11fb85ff3a1b67d9046ef101':raise ValueError('Extra division selection changed')
    private=json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    if private['movie_plan_sha256']!=sha(movie_path):raise ValueError('Archive plan mismatch')
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/dev/shm')):raise ValueError('Require new owned RAM output')
    if shutil.disk_usage('/dev/shm').free<1024**3:raise RuntimeError('Temporary RAM storage reserve unavailable')
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='enrichment_smoke_passed' or proof['contract_sha256']!=sha(ROOT/'CONTRACT.json'):raise ValueError('Exact real-image smoke required')
    selected=movie_plan['movies']
    if args.smoke:selected=[m for e in ('44b6','6bba') for m in [x for x in selected if x['embryo']==e][:2]]
    args.output.mkdir();(args.output/'data').mkdir();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8)
    start=time.monotonic();timer=threading.Timer(300 if args.smoke else 600,lambda:os._exit(124));timer.daemon=True;timer.start()
    records=[];processed=[];totals={};result=dict(run_id='native-division-v3-extra',status='running',smoke=args.smoke,
        contract_sha256=sha(ROOT/'CONTRACT.json'),records=records,processed=processed,totals=totals,
        sealed_audit_opened=False,target_pilot_labels_used=False,authorized_for_submission=False)
    archive=helper['Archive'](private);bytes_written=0
    try:
        for movie in selected:
            helper['preflight']()
            if movie['role']!='optimization':raise ValueError('Extra sampling must not change selection')
            times=movie['selected_transitions'][:1] if args.smoke else movie['selected_transitions']
            frames=sorted({f for t in times for f in (t,t+1)})
            images,metadata=archive.movie(movie,frames);node_time={int(n[0]):int(n[1]) for n in movie['nodes']}
            for t in times:
                helper['preflight']()
                parent_image,pooled=normalize_gpu(images[t],torch);parent_points=proposals(pooled)
                child_image,pooled=normalize_gpu(images[t+1],torch);child_points=proposals(pooled)
                if 'geometry_max_error' not in result:result['geometry_max_error']=helper['geometry_smoke'](parent_image)
                bindings,positions,counts=native_bindings(movie,t,parent_points,child_points)
                packet,used,extra=triplets(bindings,positions);counts.update(extra)
                degree=Counter(int(a) for a,b in movie['edges'] if node_time[int(a)]==t)
                counts['available_divisions']=sum(n==2 for n in degree.values());counts['transitions']=1
                key=movie['embryo']+'-optimization';total=totals.setdefault(key,{k:0 for k in counts})
                for k,v in counts.items():total[k]+=v
                processed.append(dict(stem=movie['stem'],transition=t,counts=counts))
                if not used:continue
                p=np.array([positions[i] for i in used if i<len(parent_points)],np.float32).reshape(-1,3)
                c=np.array([positions[i] for i in used if i>=len(parent_points)],np.float32).reshape(-1,3)
                packet['patches']=np.concatenate((patches_gpu(parent_image,p,torch),patches_gpu(child_image,c,torch)))
                if bytes_written+packet['patches'].nbytes>512*1024**2:raise RuntimeError('Extra-packet RAM cap exceeded')
                relative=f'data/{movie["stem"]}-{t:03d}.npz';destination=args.output/relative;np.savez_compressed(destination,**packet)
                size=destination.stat().st_size;bytes_written+=size
                records.append(dict(path=relative,sha256=sha(destination),bytes=size,stem=movie['stem'],embryo=movie['embryo'],role='optimization',
                                    transition=t,patches=len(used),**extra))
            del images
            result['elapsed_seconds']=time.monotonic()-start;helper['save'](args.output/'PROGRESS.json',result)
            print(json.dumps(dict(stem=movie['stem'],completed_transitions=len(processed),packets=len(records),bytes=bytes_written)),flush=True)
        if not args.smoke and len(processed)!=93:raise ValueError('Incomplete extra transition coverage')
        result['status']='enrichment_smoke_passed' if args.smoke else 'enrichment_complete'
    except Exception as error:result.update(status='failed',error_type=type(error).__name__)
    finally:
        timer.cancel();result.update(elapsed_seconds=time.monotonic()-start,bytes=bytes_written)
        helper['save'](args.output/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','processed')}),flush=True)
    return 0 if result['status'] in ('enrichment_smoke_passed','enrichment_complete') else 1


if __name__=='__main__':main()
