"""Read-only, label-free attribution of a neutral native-image graph candidate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
import threading
import os
import time
import numpy as np
import torch
import zarr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import normalize_gpu
from research.native_correspondence_models_v2 import VisualCorrespondence
from research.native_correspondence_inference_v2 import encode_native_points,score_embeddings
from research.native_graph_repair_v2 import frame_candidates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-predictions',type=Path);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/run-native-repair-pilot-v2.py'))
    if helper['gpu_pids']():raise RuntimeError('GPU occupied')
    if args.output.exists():raise ValueError('Preserve existing attribution')
    if args.cache_predictions:args.cache_predictions.mkdir(exist_ok=False)
    contract=json.loads((ROOT/'PILOT.json').read_text());models=[]
    for row in contract['files']:
        if helper['sha'](ROOT/row['path'])!=row['sha256']:raise ValueError('Pilot code/input changed')
    for row in contract['models']:
        weights=Path(row['path'])
        if helper['sha'](weights)!=row['weights_sha256']:raise ValueError('Weights changed')
        model=VisualCorrespondence(row['family'],input_channels=3).cuda().eval()
        model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True)['state_dict']);models.append(model)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    begin=time.monotonic();timer=threading.Timer(180,lambda:os._exit(124));timer.daemon=True;timer.start()
    summaries=[]
    try:
        for item in contract['movies']:
            graph=json.loads((ROOT/item['graph']).read_text());outgoing=Counter(e['source_id'] for e in graph['edges'])
            incoming={e['target_id']:e['source_id'] for e in graph['edges']}
            array=zarr.open_array(str(Path(item['images'])/'0'),mode='r');previous=None;count=Counter();maxima=[];nulls=[];free_maxima=[]
            cached_ids=[];cached_probabilities=[]
            for t in range(99):
                if helper['gpu_pids']()-{os.getpid()}:raise RuntimeError('Yield for foreign GPU')
                packet=frame_candidates(graph,t)
                if previous is None:
                    image,_=normalize_gpu(np.asarray(array[t]),torch);previous=encode_native_points(models,image,packet['parent_coords'])
                image,_=normalize_gpu(np.asarray(array[t+1]),torch);current=encode_native_points(models,image,packet['child_coords'])
                for first in range(0,len(packet['ids']),128):
                    last=min(first+128,len(packet['ids']));n=last-first
                    ids=packet['ids'][first:last];valid=torch.from_numpy(packet['valid'][first:last]).cuda();coords=torch.from_numpy(packet['coords'][first:last]).cuda()
                    probability=torch.zeros((n,17),device='cuda')
                    with torch.inference_mode():
                        for i,model in enumerate(models):
                            features=torch.zeros((n,17,256),device='cuda',dtype=torch.float16);features[:,0]=current[i][first:last].cuda()
                            if len(packet['parent_node_ids']):features[:,1:]=previous[i][packet['parent_indices'][first:last]].cuda()
                            with torch.autocast('cuda',dtype=torch.float16):logits=score_embeddings(model,features,coords,valid)
                            probability+=contract['models'][i]['weight']*torch.softmax(logits.float(),dim=-1)
                    p=probability.cpu().numpy();best=p.argmax(-1);maxima.extend(p.max(-1).tolist());nulls.extend(p[:,-1].tolist())
                    if args.cache_predictions:cached_ids.append(ids.copy());cached_probabilities.append(p.copy())
                    count['queries']+=n;count['top_null']+=int((best==16).sum());count['top_probability_ge_099']+=int((p.max(-1)>=.99).sum())
                    for row in range(n):
                        child=int(ids[row,0]);old=incoming.get(child)
                        free=[k for k,parent in enumerate(ids[row,1:]) if parent>=0 and outgoing[int(parent)]==0]
                        if old is None:
                            count['unlinked_child']+=1
                            if free:count['unlinked_child_with_free_candidate']+=1;free_maxima.append(float(p[row,free].max()))
                        if best[row]==16:continue
                        parent=int(ids[row,best[row]+1]);count['top_agrees_existing_edge']+=int(parent==old)
                        if p[row,best[row]]<.99:continue
                        count['confident_real_parent']+=1
                        if parent==old:count['confident_agrees_existing_edge']+=1;continue
                        if outgoing[parent]!=0:count['confident_other_parent_occupied']+=1;continue
                        if old is not None and outgoing[old]!=1:count['confident_blocked_existing_division']+=1;continue
                        if old is not None and old not in ids[row,1:]:count['confident_old_parent_outside_context']+=1;continue
                        count['eligible_frozen_repair']+=1
                previous=current
            quantiles=lambda values:np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist() if values else []
            row=dict(stem=item['stem'],counts=dict(count),quantile_levels=[0,.25,.5,.75,.9,.99,1],
                     top_probability_quantiles=quantiles(maxima),null_probability_quantiles=quantiles(nulls),
                     unlinked_child_free_parent_probability_quantiles=quantiles(free_maxima))
            if args.cache_predictions:
                destination=args.cache_predictions/(item['stem']+'.npz')
                np.savez_compressed(destination,ids=np.concatenate(cached_ids),probabilities=np.concatenate(cached_probabilities))
                row['prediction_cache']=dict(file=destination.name,sha256=helper['sha'](destination),bytes=destination.stat().st_size)
            summaries.append(row);print(json.dumps(row),flush=True)
        result=dict(status='label_free_attribution_complete',movies=summaries,elapsed_seconds=time.monotonic()-begin,
                    pilot_sha256=helper['sha'](ROOT/'PILOT.json'),source_sha256=helper['sha'](Path(__file__)),
                    ground_truth_opened=False,model_or_graph_changed=False,authorized_for_submission=False)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:timer.cancel()


if __name__=='__main__':main()
