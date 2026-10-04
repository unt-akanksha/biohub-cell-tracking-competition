"""Bounded real-image forward/backward/reload/timing; no selection evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from research.native_division_context_model_v6 import TemporalDivision,paired_crops


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cp=ROOT/'CONTRACT.json';contract=json.loads(cp.read_text())
    for r in contract['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Model smoke code changed')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(int(line)!=os.getpid() for line in active.splitlines() if line.strip()):raise RuntimeError('Foreign GPU process; do not touch')
    manifest=json.loads((args.data/'RESULT.json').read_text())
    if manifest['status']!='context_smoke_passed' or manifest['contract_sha256']!='79b8b979e2b4b3c1189b8e362ffb0f60b1b4f17201dd8442d56b4d3896032acd':raise ValueError('Verified data smoke required')
    positive=[];negative=[]
    for r in manifest['records']:
        if r['role']!='optimization':raise ValueError('No selection in model smoke')
        path=args.data/r['path']
        if sha(path)!=r['sha256']:raise ValueError('Data changed')
        with np.load(path,allow_pickle=False) as p:
            for row,y in enumerate(p['labels']):
                ids=p['triples'][row]
                sample=(p['patches'][ids],p['context_patches'][ids],p['coords'][row],p['context_valid'][ids],int(y))
                (positive if y else negative).append(sample)
    # Real smoke includes both source embryos, but cannot create training output.
    batch=[positive[i%len(positive)] for i in range(4)]+[negative[i%len(negative)] for i in range(4)]
    original,context,coords,valid,y=[torch.from_numpy(np.array([r[k] for r in batch])).cuda() for k in range(5)]
    coords=coords.float();y=y.float();torch.set_num_threads(2);torch.manual_seed(9601);torch.backends.cudnn.benchmark=False
    model=TemporalDivision().cuda()
    weight=Path('/tmp/biohub-image-context-v2.ScdSdY/native-correspondence-v2-training-full/6bba-resnet3d/best.pt')
    if sha(weight)!='6fbd68f59d68cb7970f8e11c87d3e171243bdfd447c70a9fe13ce15dd2500f73':raise ValueError('Source encoder changed')
    model.load_source_encoder(torch.load(weight,map_location='cuda',weights_only=True)['state_dict'])
    # Nonzero output weights allow a meaningful encoder-gradient and symmetry probe.
    torch.nn.init.normal_(model.event[-1].weight,std=.01)
    generator=torch.Generator(device='cuda').manual_seed(9601)
    a,c,xyz=paired_crops(original,context,coords,valid,augment=True,generator=generator)
    identical,identical_context,_=paired_crops(original,original,coords,torch.ones_like(valid),augment=True,generator=generator)
    torch.testing.assert_close(identical,identical_context,rtol=0,atol=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    scaler=torch.amp.GradScaler('cuda');timings=[]
    for step in range(25):
        start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(a,c,xyz,valid);loss=F.binary_cross_entropy_with_logits(logits,y)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        timings.append(time.perf_counter()-start)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        expected=model(a,c,xyz,valid)
        swapped=model(a[:,[0,2,1]],c[:,[0,2,1]],xyz[:,[0,2,1]],valid[:,[0,2,1]])
        torch.testing.assert_close(expected,swapped,rtol=0,atol=1e-5)
        masked=model(a,c,xyz,torch.zeros_like(valid))
        no_context=model(a,c*0,xyz,valid,use_context=False)
        torch.testing.assert_close(masked,no_context,rtol=0,atol=0)
    args.output.mkdir(exist_ok=False);path=args.output/'smoke.pt';torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):torch.testing.assert_close(expected,model(a,c,xyz,valid),rtol=0,atol=0)
    encoder_grad=sum(p.numel() for p in model.encoder.parameters() if p.grad is not None and bool((p.grad!=0).any()))
    if encoder_grad==0:raise ValueError('Encoder did not receive gradients')
    result=dict(status='temporal_model_smoke_passed',contract_sha256=sha(cp),steps=25,batch_size=8,
                parameters=sum(p.numel() for p in model.parameters()),encoder_parameters_with_gradient=encoder_grad,
                median_step_seconds=float(np.median(timings[5:])),last_loss=float(loss),exact_reload=True,
                daughter_swap_max_error=float((expected-swapped).abs().max()),boundary_mask_exact=True,
                weights_sha256=sha(path),source_only_training=False,authorized_for_training_initialization=False,
                authorized_for_submission=False)
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
