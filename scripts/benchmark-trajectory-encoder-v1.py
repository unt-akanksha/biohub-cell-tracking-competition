"""Image-only sequential A10 benchmark: batching/precision, no weight updates."""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser()
    for key in ('bundle','images','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    start=time.monotonic();result=dict(status='initializing',ground_truth_opened=False,
        weights_changed=False,authorized_for_submission=False,cases=[])
    def save():
        result['elapsed_seconds']=time.monotonic()-start
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    def timeout():
        result['status']='timeout';save();os._exit(124)
    timer=threading.Timer(1200,timeout);timer.daemon=True;timer.start()
    try:
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                             capture_output=True,text=True,check=True,timeout=15).stdout
        if query.strip():raise ValueError('Shared GPU occupied; no launch permitted')
        contract=args.bundle/'CONTRACT.json'
        if sha(contract)!='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1':
            raise ValueError('Frozen Antelume runtime changed')
        pins=json.loads(contract.read_text())['bundle_sha256']
        for name,value in pins.items():
            if sha(args.bundle/name)!=value:raise ValueError('Runtime input changed')
        os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        import numpy as np
        import torch
        import zarr
        if torch.cuda.device_count()!=1:raise ValueError('One free shared GPU required')
        torch.set_num_threads(2);torch.manual_seed(1729)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.80)
        pre=load(args.bundle/'public_d4_preflight.py','benchmark_pre')
        temporal=load(args.bundle/'temporal_unet.py','benchmark_temporal')
        # Per-voxel temporal sequences are independent. Bound the flattened
        # batch dimension below the torch2.5 SDPA CUDA grid limit (issue142228).
        # This adapter is benchmark-only; production runtime is untouched.
        def chunked_temporal(self,x):
            b,t,c=x.shape[:3];spatial=x.shape[3:];s=math.prod(spatial)
            h=x.reshape(b,t,c,s).permute(0,3,1,2).reshape(b*s,t,c)
            h=self.norm(h)
            h=torch.cat([self.attn(part,part,part,need_weights=False)[0]
                         for part in h.split(32768,dim=0)],dim=0)
            return x+h.reshape(b,s,t,c).permute(0,2,3,1).reshape(b,t,c,*spatial)
        temporal._TemporalAttention.forward=chunked_temporal
        transformer=load(args.bundle/'simple_node_transformer.py','benchmark_transformer')
        env=dict(torch=torch,nn=torch.nn,np=np,_POS_EMBED_DIM=8,
                 SimpleNodeTransformer=transformer.SimpleNodeTransformer)
        exec(compile(pre.named_definitions((args.bundle/'train_unet_transformer.py').read_text(),
            ('UNetNodeTransformer','extract_pos_features')),'pinned-model-definitions','exec'),env)
        result.update(status='running',device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                      source_contract_sha256=sha(contract),source_script_sha256=sha(Path(__file__)),
                      benchmark_revision=2,temporal_attention_batch_chunk=32768)
        def encode(model,batch,dtype):
            ctx=torch.autocast('cuda',dtype=torch.float16) if dtype=='fp16' else contextlib.nullcontext()
            with ctx:
                features,logits=model.encode(batch)
            return features,torch.cat(logits,dim=1)
        for model_name in ('primary','secondary'):
            model=env['UNetNodeTransformer'](temporal.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),
                                            unet_out_channels=32,pos_feat_dim=32)
            model.load_state_dict(torch.load(args.bundle/(model_name+'.pth'),map_location='cpu',weights_only=True),strict=True)
            model=model.cuda().eval()
            for stem in ('44b6_12dfb391','6bba_07e24132'):
                image=args.images/'train'/(stem+'.zarr')
                attrs=json.loads((image/'zarr.json').read_text())['attributes']
                quantiles=attrs['image_statistics']['quantiles']
                low,high=float(quantiles['0.001']),float(quantiles['0.999'])
                array=zarr.open(str(image/'0'),mode='r')
                frames=torch.stack([torch.from_numpy(np.asarray(array[t,:,::4,::4],dtype=np.float32)) for t in range(17)])
                frames=((frames-low)/(high-low+1e-6)).clamp(0.)
                inputs=torch.stack([frames[t:t+2] for t in range(16)]).cuda()
                if inputs.shape!=(16,2,64,64,64):raise ValueError('Unexpected image geometry')
                with torch.inference_mode():
                    # Functionality smoke always precedes timed/bigger batches.
                    smoke_f,smoke_l=encode(model,inputs[:1],'fp32')
                    if not torch.isfinite(smoke_f).all() or not torch.isfinite(smoke_l).all():
                        raise ValueError('Nonfinite baseline smoke')
                    smoke_f,smoke_l=encode(model,inputs[:2],'fp16')
                    if not torch.isfinite(smoke_f).all() or not torch.isfinite(smoke_l).all():
                        raise ValueError('Nonfinite mixed-precision smoke')
                    del smoke_f,smoke_l
                    print(json.dumps(dict(event='smoke_passed',model=model_name,movie=stem)),flush=True)
                    transforms=(lambda x:x,lambda x:x.flip((-1,)),lambda x:x.transpose(-1,-2))
                    for view,transform in enumerate(transforms):
                        batch=transform(inputs)
                        references=[]
                        for j in range(16):
                            f,l=encode(model,batch[j:j+1],'fp32');references.append((f.detach(),l.detach()))
                            del f,l
                        for dtype,bs in (('fp32',1),('fp32',4),('fp32',8),('fp32',16),('fp16',1),('fp16',4),('fp16',8),('fp16',16)):
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
                            max_feature_error=0.;max_logit_error=0.;mean_logit_error=0.;mismatches=0;gpu_seconds=0.
                            for j in range(0,16,bs):
                                event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True)
                                event0.record()
                                f,l=encode(model,batch[j:j+bs],dtype)
                                event1.record();event1.synchronize();gpu_seconds+=event0.elapsed_time(event1)/1000
                                f,l=f.float(),l.float()
                                for k in range(bs):
                                    rf,rl=references[j+k]
                                    max_feature_error=max(max_feature_error,float((f[k:k+1]-rf).abs().max()))
                                    error=(l[k:k+1]-rl).abs()
                                    max_logit_error=max(max_logit_error,float(error.max()))
                                    mean_logit_error+=float(error.mean())/16
                                    mismatches+=int(((l[k:k+1].sigmoid()>.965)!=(rl.sigmoid()>.965)).sum())
                                del f,l
                            torch.cuda.synchronize()
                            row=dict(model=model_name,movie=stem,view=view,dtype=dtype,batch_size=bs,
                                     windows=16,seconds=time.monotonic()-t0,gpu_encode_seconds=gpu_seconds,
                                     peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                                     max_feature_error=max_feature_error,max_logit_error=max_logit_error,
                                     mean_logit_error=mean_logit_error,voxel_threshold_disagreements=mismatches,
                                     comparison_on_gpu=True)
                            result['cases'].append(row);save();print(json.dumps(row),flush=True)
                        del references,batch
                del inputs,frames
                gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
        if any(sha(args.bundle/name)!=value for name,value in pins.items()):raise ValueError('Inputs changed during benchmark')
        result.update(status='complete',inputs_unchanged=True,quality_promotion_permitted=False)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        timer.cancel();save();print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
