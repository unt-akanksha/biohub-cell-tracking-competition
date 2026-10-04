"""One frozen cross-embryo diagnostic; never optimize against target outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_correspondence_models_v2 import VisualCorrespondence, mask_scores


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); helpers=runpy.run_path(str(ROOT/'scripts/train-native-correspondence-v2.py'))
    sha=helpers['sha']; save=helpers['save']; pids=helpers['pids']; load=helpers['load_data']; batch=helpers['batch']
    if pids(): raise RuntimeError('GPU occupied')
    if args.output.exists(): raise ValueError('Preserve diagnostic result')
    training=json.loads((args.training/'RESULT.json').read_text()); data=json.loads((args.data/'RESULT.json').read_text())
    if training['status']!='training_complete' or training['data_result_sha256']!=sha(args.data/'RESULT.json'):
        raise ValueError('Require exact completed training/data')
    contract=json.loads((ROOT/'TRAINING.json').read_text())
    if training['training_contract_sha256']!=sha(ROOT/'TRAINING.json'): raise ValueError('Training contract mismatch')
    for record in contract['files']:
        if sha(ROOT/record['path'])!=record['sha256']: raise ValueError('Training source mismatch')
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.9)
    begin=time.monotonic(); result=dict(run_id='native-correspondence-v2',folds=[],authorized_for_submission=False,
        training_result_sha256=sha(args.training/'RESULT.json'),data_result_sha256=sha(args.data/'RESULT.json'),
        mixture_weights=[.5,.5],script_sha256=sha(Path(__file__)))
    @torch.inference_mode()
    def probabilities(model,packet,missing=False):
        if model is not None: model.eval()
        output=[]; targets=[]
        for first in range(0,len(packet['targets']),12):
            rows=torch.arange(first,min(first+12,len(packet['targets'])),device='cuda')
            x,coord,valid,mask,target=batch(packet,rows,missing=missing)
            if model is None:
                distance=((coord[:,1:]-coord[:,:1])/10).square().sum(-1)
                raw=torch.cat((-2*distance,torch.full_like(distance[:,:1],-8)),dim=-1)
                raw=raw.masked_fill(~torch.cat((valid[:,1:],valid[:,:1]),dim=-1),-1e4)
            else:
                with torch.autocast('cuda',dtype=torch.float16): raw=model(x,coord,valid)
            output.append(torch.softmax(mask_scores(raw,mask).float(),-1).cpu()); targets.append(target.cpu())
        return torch.cat(output),torch.cat(targets)
    def metrics(probability,target,movie,names):
        loss=-probability[torch.arange(len(target)),target].clamp_min(1e-30).log()
        correct=probability.argmax(-1)==target
        return dict(nll=float(loss.mean()),correct=int(correct.sum()),total=len(target),per_movie={name:dict(
            nll=float(loss[movie==i].mean()),correct=int(correct[movie==i].sum()),total=int((movie==i).sum())) for i,name in enumerate(names)})
    for source,target_embryo in [('6bba','44b6'),('44b6','6bba')]:
        members=[m for m in training['members'] if m['embryo']==source]
        fold=dict(source=source,target=target_embryo,source_gate={m['family']:m['source_gate'] for m in members})
        admitted=[m for m in members if m['source_gate']]
        if not admitted:
            fold.update(status='target_not_opened_source_failure'); result['folds'].append(fold); continue
        packet=load(args.data,data['records'],target_embryo,'selection')
        movie=packet['movie_ids'].cpu(); names=packet['movie_names']
        baseline,label=probabilities(None,packet); absent,missing_label=probabilities(None,packet,True)
        fold.update(baseline=metrics(baseline,label,movie,names),missing_baseline=metrics(absent,missing_label,movie,names),methods={})
        probs=[]; missing_probs=[]
        for member in admitted:
            weights=args.training/(source+'-'+member['family'])/'best.pt'
            if sha(weights)!=member['weights_sha256']: raise ValueError('Selected weights changed')
            model=VisualCorrespondence(member['family'],input_channels=3).cuda()
            checkpoint=torch.load(weights,map_location='cuda',weights_only=True); model.load_state_dict(checkpoint['state_dict'])
            p,y=probabilities(model,packet); pm,ym=probabilities(model,packet,True)
            if not torch.equal(y,label) or not torch.equal(ym,missing_label): raise ValueError('Evaluation order changed')
            probs.append(p); missing_probs.append(pm)
            fold['methods'][member['family']]=dict(real=metrics(p,label,movie,names),missing=metrics(pm,missing_label,movie,names))
            del model,checkpoint; torch.cuda.empty_cache()
        if len(probs)==2:
            fold['methods']['fixed_equal_mixture']=dict(real=metrics((probs[0]+probs[1])*.5,label,movie,names),
                                missing=metrics((missing_probs[0]+missing_probs[1])*.5,missing_label,movie,names))
            a=probs[0].argmax(-1)==label; b=probs[1].argmax(-1)==label
            fold['complementarity']=dict(first_only=int((a&~b).sum()),second_only=int((b&~a).sum()),both_wrong=int((~a&~b).sum()))
        for method in fold['methods'].values():
            method['conditional_gate']=method['real']['correct']>=fold['baseline']['correct'] and method['real']['nll']<fold['baseline']['nll'] and method['missing']['nll']<=fold['missing_baseline']['nll']
        fold['status']='evaluated_once'; result['folds'].append(fold)
        del packet; torch.cuda.empty_cache()
    result.update(status='diagnostic_complete',elapsed_seconds=time.monotonic()-begin)
    save(args.output,result); print(json.dumps(result))


if __name__=='__main__': main()
