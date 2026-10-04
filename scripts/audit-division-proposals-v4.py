"""Source-only native proposal coverage, not target validation or fitting."""
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
from research.native_correspondence_data_v2 import normalize_gpu,proposals
from research.native_division_proposal_audit_v4 import coverage,proposals_xy2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-proof',type=Path);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'));helper['preflight']()
    contract_path=ROOT/'CONTRACT.json';contract=json.loads(contract_path.read_text())
    for r in contract['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Frozen audit changed')
    plan_path=ROOT/'plan/MOVIES.json';plan=json.loads(plan_path.read_text())
    archive_plan=json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    if archive_plan['movie_plan_sha256']!=sha(plan_path) or plan['total_transitions']!=112:raise ValueError('Wrong source scope')
    if args.output.exists():raise ValueError('Do not overwrite an audit')
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='proposal_audit_smoke_passed' or proof['contract_sha256']!=sha(contract_path):raise ValueError('Matching smoke required')
    movies=plan['movies']
    if args.smoke:movies=[m for e in ('44b6','6bba') for m in [m for m in movies if m['embryo']==e][:2]]
    args.output.mkdir();(args.output/'points').mkdir()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    started=time.monotonic();timer=threading.Timer(600,lambda:os._exit(124));timer.daemon=True;timer.start()
    archive=helper['Archive'](archive_plan);records=[];totals={};artifacts=[];frames=[]
    result=dict(status='running',contract_sha256=sha(contract_path),movie_plan_sha256=sha(plan_path),records=records,
                totals=totals,point_artifacts=artifacts,frames=frames,selection_opened=False,target_pilot_labels_used=False,
                authorized_for_submission=False)
    try:
        for movie in movies:
            helper['preflight']()
            if movie['role']!='optimization':raise ValueError('Only optimization division events allowed')
            times=movie['selected_transitions'][:1] if args.smoke else movie['selected_transitions']
            selected_frames=sorted({f for t in times for f in (t,t+1)})
            images,metadata=archive.movie(movie,selected_frames);points={};packet={}
            for t in selected_frames:
                helper['preflight']();start=time.monotonic()
                normalized,pooled=normalize_gpu(images[t],torch)
                baseline=proposals(pooled);baseline_seconds=time.monotonic()-start
                start=time.monotonic()
                half=normalized.reshape(64,128,2,128,2).mean((2,4)).cpu().numpy()
                fine=proposals_xy2(half);fine_seconds=time.monotonic()-start
                points[t]=dict(baseline=baseline,xy2=fine)
                for variant,array in points[t].items():packet[f'{variant}_{t:03d}']=array
                frames.append(dict(stem=movie['stem'],time=t,raw_sha256=hashlib.sha256(images[t].tobytes()).hexdigest(),
                                   baseline_points=len(baseline),xy2_points=len(fine),baseline_with_normalization_seconds=baseline_seconds,
                                   xy2_after_normalization_seconds=fine_seconds))
            for t in times:
                for variant in ('baseline','xy2'):
                    evaluated=coverage(movie,t,points[t][variant],points[t+1][variant])
                    records.append(dict(stem=movie['stem'],embryo=movie['embryo'],transition=t,variant=variant,**evaluated))
                    total=totals.setdefault(movie['embryo']+'-'+variant,{})
                    for key,value in evaluated['counts'].items():total[key]=total.get(key,0)+value
            path=args.output/'points'/(movie['stem']+'.npz');np.savez_compressed(path,**packet)
            artifacts.append(dict(path=path.relative_to(args.output).as_posix(),bytes=path.stat().st_size,sha256=sha(path),metadata=metadata))
            del images,points,normalized
            helper['save'](args.output/'PROGRESS.json',result)
            print(json.dumps(dict(stem=movie['stem'],transitions=len(records)//2,seconds=time.monotonic()-started)),flush=True)
        if not args.smoke and len(records)!=224:raise ValueError('Incomplete full source audit')
        result['status']='proposal_audit_smoke_passed' if args.smoke else 'proposal_audit_complete'
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__)
    finally:
        timer.cancel();result['seconds']=time.monotonic()-started
        helper['save'](args.output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','frames','point_artifacts')}),flush=True)
    return 0 if result['status'] in ('proposal_audit_smoke_passed','proposal_audit_complete') else 1


if __name__=='__main__':raise SystemExit(main())
