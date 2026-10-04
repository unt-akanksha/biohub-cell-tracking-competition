"""Known-deformation linker smoke; mixed-source weights never qualify for use."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.dense_warp_correspondence_v1 import forward_points, inverse_grid, known_pairs


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('transformer.'): continue
        digest.update(name.encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    helper = runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'))
    helper['preflight']()
    assert args.output.parent == Path('/dev/shm') and args.output.name.startswith('biohub-dense-warp-') and not args.output.exists()
    contract = json.loads((ROOT/'CONTRACT.json').read_text())
    for row in contract['files']: assert sha(ROOT/row['path']) == row['sha256']
    plan = json.loads((ROOT/'plan/MOVIES.json').read_text())
    private = json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    assert private['movie_plan_sha256'] == sha(ROOT/'plan/MOVIES.json')
    assert not plan['ground_truth_used'] and all(m['role']=='optimization' for m in plan['movies'])
    args.output.mkdir()
    result = dict(status='running', contract_sha256=sha(ROOT/'CONTRACT.json'),
                  source_only_functionality=True, ground_truth_used=False,
                  authorized_for_submission=False, model_eligible_for_initialization=False)
    started = time.monotonic()
    timer = threading.Timer(600, lambda: os._exit(124)); timer.daemon=True; timer.start()
    try:
        torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.65)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(9721); rng = np.random.default_rng(9721)
        model_helpers = load(ROOT/'public/public_d4_preflight.py', 'dense_warp_helpers')
        temporal = load(ROOT/'public/temporal_unet.py', 'dense_warp_temporal')
        transformer = load(ROOT/'public/simple_node_transformer.py', 'dense_warp_transformer')
        namespace = dict(torch=torch, nn=torch.nn, np=np, SimpleNodeTransformer=transformer.SimpleNodeTransformer)
        definitions = model_helpers.named_definitions((ROOT/'public/train_unet_transformer.py').read_text(), ('UNetNodeTransformer','extract_pos_features'))
        namespace['_POS_EMBED_DIM'] = 8
        exec(compile(definitions, 'audited-public-model-definitions', 'exec'), namespace)
        def make_model():
            return namespace['UNetNodeTransformer'](temporal.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda().eval()
        model = make_model()
        model.load_state_dict(torch.load(ROOT/'public/primary.pth', map_location='cpu', weights_only=True), strict=True)
        for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith('transformer.'))
        frozen = frozen_sha(model)
        # Integer translation must move a real impulse to the transformed point.
        impulse = torch.zeros(1,1,64,64,64,device='cuda'); impulse[0,0,20,22,24] = 1
        translated = F.grid_sample(impulse, torch.from_numpy(inverse_grid([1,2,-3,0,0]))[None].cuda(),align_corners=True)
        expected = forward_points(np.array([[20,22,24]]), [1,2,-3,0,0])[0].astype(int)
        observed = np.array(np.unravel_index(int(translated.argmax()), (64,64,64)))
        assert np.array_equal(expected, observed) and abs(float(translated.max())-1)<1e-5
        result['image_point_translation_exact'] = True
        archive = helper['Archive'](private); packets=[]; frames={}; rows=[]
        for movie in plan['movies']:
            helper['preflight'](); t = movie['image_frames'][0]
            images, metadata = archive.movie(movie, [t]); raw = images[t]
            group = json.loads(archive.member('train/'+movie['stem']+'.zarr/zarr.json'))
            q = group['attributes']['image_statistics']['quantiles']
            volume = np.maximum((raw[:,::4,::4].astype(np.float32)-q['0.001'])/(q['0.999']-q['0.001']+1e-6),0)
            image = torch.from_numpy(volume).cuda()
            frames[movie['stem']] = raw
            with torch.no_grad():
                _, logits = model.encode(torch.stack((image,image))[None])
                score=logits[0]; peak=(score==F.max_pool3d(score,3,1,1)) & (score.sigmoid()>.965)
                source = peak[0,0].nonzero().float().cpu().numpy()
            source = source[((source>=2)&(source<=61)).all(-1)]
            if not 8 <= len(source) <= 1536: raise ValueError('Dense source peak count outside smoke budget; do not truncate')
            for view in range(2):
                params = np.r_[rng.uniform(-3,3,3),rng.uniform(-2,2,2)]
                target, labels = known_pairs(source, params, rng.permutation(len(source)))
                assert len(target)>=8
                grid = torch.from_numpy(inverse_grid(params))[None].cuda()
                with torch.no_grad():
                    warped = F.grid_sample(image[None,None],grid,align_corners=True)[0,0]
                    warped = warped * float(rng.uniform(.85,1.15))
                    features, _ = model.encode(torch.stack((image,warped))[None])
                    coords = [torch.from_numpy(c)[None].cuda() for c in (source,target)]
                    masks = [torch.ones(1,len(c),device='cuda',dtype=torch.bool) for c in (source,target)]
                    indexed = [model._index_features(features[:,i],coords[i],masks[i]) for i in range(2)]
                    pos = [torch.from_numpy(namespace['extract_pos_features'](np.column_stack((np.full(len(c),i),c)),(2,64,64,64)))[None].cuda() for i,c in enumerate((source,target))]
                inputs = (*indexed,*(c*torch.tensor([1,4,4],device='cuda') for c in coords),*pos,*masks)
                packets.append((inputs,torch.from_numpy(labels).cuda()))
                rows.append(dict(stem=movie['stem'],time=t,view=view,source=len(source),target=len(target),parameters=params.tolist(),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),**metadata))
                del features, warped
        np.savez_compressed(args.output/'source-frames.npz',**frames)
        def evaluate(net):
            net.eval(); correct=total=0; losses=[]
            with torch.no_grad():
                for inputs, labels in packets:
                    logits=net.predict_edges(*inputs)[0].T
                    assert torch.isfinite(logits).all()
                    losses.append(float(F.cross_entropy(logits,labels)))
                    correct += int((logits.argmax(-1)==labels).sum()); total += len(labels)
            return dict(correct=correct,total=total,mean_pair_ce=float(np.mean(losses)))
        result['before'] = evaluate(model)
        optimizer = torch.optim.AdamW(model.transformer.parameters(),lr=1e-4,weight_decay=.01)
        model.transformer.train(); history=[]; gradients=set(); train_start=time.monotonic()
        for step in range(25):
            helper['preflight']()
            inputs,labels=packets[step%len(packets)]
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model.predict_edges(*inputs)[0].T,labels)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.transformer.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                if bool((p.grad!=0).any()): gradients.add(name)
            norm=torch.nn.utils.clip_grad_norm_(model.transformer.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); history.append(dict(step=step+1,loss=float(loss),grad_norm=float(norm)))
        result['training_seconds']=time.monotonic()-train_start
        assert gradients==set(dict(model.transformer.named_parameters()))
        assert frozen_sha(model)==frozen
        result['after']=evaluate(model)
        checkpoint=args.output/'smoke-only.pt'
        torch.save(dict(state_dict=model.state_dict(),smoke_only=True,eligible_for_initialization=False,seed=9721),checkpoint)
        restored=make_model(); restored.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'],strict=True)
        with torch.no_grad():
            error=float((restored.predict_edges(*packets[0][0])-model.predict_edges(*packets[0][0])).abs().max())
        assert error==0
        result.update(status='dense_warp_smoke_passed',rows=rows,history=history,
                      frozen_modules_sha256=frozen,linker_parameters=sum(p.numel() for p in model.transformer.parameters()),
                      nonzero_gradient_tensors=len(gradients),reload_max_error=error,model_sha256=sha(checkpoint),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),steps=25,
                      source_frames_sha256=sha(args.output/'source-frames.npz'))
    except Exception as error:
        # Do not print exception text: a network error can include a signed URL.
        result.update(status='failed',error_type=type(error).__name__)
    finally:
        timer.cancel();result['seconds']=time.monotonic()-started
        helper['save'](args.output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('history','rows')}),flush=True)
    return 0 if result['status']=='dense_warp_smoke_passed' else 1


if __name__=='__main__':raise SystemExit(main())
