"""Prelabel native-image repairs on four full, previously excluded movies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import torch
import zarr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import normalize_gpu
from research.native_correspondence_models_v2 import VisualCorrespondence
from research.native_correspondence_inference_v2 import encode_native_points,score_embeddings
from research.native_graph_repair_v2 import frame_candidates,propose_repairs,apply_repairs
from research.trajectory_runtime_v1 import validate_graph,image_metadata


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_pids():
    output=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
    return {int(s) for s in output.splitlines() if s.strip()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-proof',type=Path)
    args=parser.parse_args();contract=json.loads((ROOT/'PILOT.json').read_text())
    for record in contract['files']:
        if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Pilot source/input changed')
    if gpu_pids():raise RuntimeError('GPU occupied')
    if args.output.exists():raise ValueError('Preserve previous output')
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='native_repair_smoke_passed' or proof['pilot_sha256']!=sha(ROOT/'PILOT.json'):
            raise ValueError('Exact native repair smoke required')
        if proof['projected_four_movie_seconds']>1500:raise RuntimeError('Pilot exceeds runtime budget')
    image_manifest=json.loads((ROOT/'IMAGE_MANIFEST.json').read_text())
    for record in image_manifest['records']:
        path=Path(contract['image_root'])/record['path']
        if not path.resolve().is_relative_to(Path(contract['image_root']).resolve()) or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Frozen native image input changed')
    args.output.mkdir();begin=time.monotonic();result=dict(status='running',pilot_sha256=sha(ROOT/'PILOT.json'),
        smoke=args.smoke,movies=[],ground_truth_opened=False,authorized_for_submission=False)
    stop=threading.Event()
    def monitor():
        while not stop.wait(15):
            try:
                if gpu_pids()-{os.getpid()}:stop.set()
            except Exception:stop.set()
    threading.Thread(target=monitor,daemon=True).start()
    timer=threading.Timer(300 if args.smoke else 1800,lambda:os._exit(124));timer.daemon=True;timer.start()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8);torch.backends.cudnn.benchmark=False
    models=[]
    try:
        for row in contract['models']:
            path=Path(row['path'])
            if sha(path)!=row['weights_sha256']:raise ValueError('Native weights changed')
            model=VisualCorrespondence(row['family'],input_channels=3).cuda().eval()
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict']);models.append(model)
        for item in contract['movies'][:2] if args.smoke else contract['movies']:
            tick=time.monotonic();stem=item['stem'];graph=json.loads((ROOT/item['graph']).read_text());validate_graph(graph,100)
            image_root=Path(item['images']);array=zarr.open_array(str(image_root/'0'),mode='r')
            image_metadata(image_root)
            if array.shape!=(100,64,256,256) or array.dtype!=np.dtype('uint16'):raise ValueError('Raw image contract changed')
            proposals=[];previous=None;encoded_nodes=0;groups=0
            transitions=range(3) if args.smoke else range(99)
            for t in transitions:
                if stop.is_set():raise RuntimeError('Yield to foreign GPU process')
                packet=frame_candidates(graph,t)
                if previous is None:
                    normalized,_=normalize_gpu(np.asarray(array[t]),torch)
                    previous=encode_native_points(models,normalized,packet['parent_coords']);encoded_nodes+=len(packet['parent_coords'])
                normalized,_=normalize_gpu(np.asarray(array[t+1]),torch)
                current=encode_native_points(models,normalized,packet['child_coords']);encoded_nodes+=len(packet['child_coords'])
                for first in range(0,len(packet['ids']),128):
                    last=min(first+128,len(packet['ids']));n=last-first
                    coords=torch.from_numpy(packet['coords'][first:last]).cuda();valid=torch.from_numpy(packet['valid'][first:last]).cuda()
                    probability=torch.zeros((n,17),device='cuda')
                    for model_index,model in enumerate(models):
                        features=torch.zeros((n,17,256),dtype=torch.float16,device='cuda')
                        features[:,0]=current[model_index][first:last].cuda()
                        if len(packet['parent_node_ids']):features[:,1:]=previous[model_index][packet['parent_indices'][first:last]].cuda()
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                            logits=score_embeddings(model,features,coords,valid)
                        probability+=contract['models'][model_index]['weight']*torch.softmax(logits.float(),dim=-1)
                    subset={k:v[first:last] for k,v in packet.items() if k in ('ids','valid')}
                    proposals.extend(propose_repairs(graph,subset,probability.cpu().numpy()));groups+=n
                previous=current
            torch.cuda.synchronize();repaired,accepted=apply_repairs(graph,proposals);validate_graph(repaired,100)
            if repaired['nodes']!=graph['nodes']:raise ValueError('Node set changed')
            # Smoke output remains explicitly partial-change diagnostic, never
            # eligible for official scoring despite retaining the full graph.
            output=args.output/(stem+'.json');output.write_text(json.dumps(repaired))
            details=args.output/(stem+'-repairs.json');details.write_text(json.dumps(accepted,indent=2)+'\n')
            record=dict(stem=stem,frames_evaluated=len(transitions)+1,groups=groups,encoded_nodes=encoded_nodes,
                        repairs=len(accepted),additions=sum(r['kind']=='addition' for r in accepted),
                        rewires=sum(r['kind']=='rewire' for r in accepted),elapsed_seconds=time.monotonic()-tick,
                        prediction_sha256=sha(output),repairs_sha256=sha(details),parent_sha256=sha(ROOT/item['graph']))
            result['movies'].append(record);print(json.dumps(record),flush=True)
            (args.output/'PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n')
        result['status']='native_repair_smoke_passed' if args.smoke else 'complete_prelabel_predictions'
        if args.smoke:
            result['projected_four_movie_seconds']=sum(m['elapsed_seconds']*100/m['frames_evaluated'] for m in result['movies'])*2*1.5
    except Exception as error:result.update(status='failed',error_type=type(error).__name__)
    finally:
        stop.set();timer.cancel();result['elapsed_seconds']=time.monotonic()-begin
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if result['status'] in ('native_repair_smoke_passed','complete_prelabel_predictions') else 1


if __name__=='__main__':raise SystemExit(main())
