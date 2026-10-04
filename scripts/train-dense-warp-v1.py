"""Source-specific dense augmentation fits, then frozen real-pair screening."""
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
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.dense_warp_correspondence_v1 import inverse_grid,known_pairs
from research.native_correspondence_data_v2 import match_queries,VOXEL


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'));helper['preflight']()
    smoke=runpy.run_path(str(ROOT/'scripts/smoke-dense-warp-v1.py'))
    load=smoke['load'];frozen_sha=smoke['frozen_sha']
    contract=json.loads((ROOT/'CONTRACT.json').read_text())
    for row in contract['files']:assert sha(ROOT/row['path'])==row['sha256']
    proof=json.loads((ROOT/'SMOKE.json').read_text())
    assert proof['status']=='dense_warp_smoke_passed' and proof['steps']==25 and proof['reload_max_error']==0
    plan=json.loads((ROOT/'plan/MOVIES.json').read_text());private=json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    assert private['movie_plan_sha256']==sha(ROOT/'plan/MOVIES.json')
    assert args.output.parent==Path('/dev/shm') and args.output.name=='biohub-dense-warp-v1-full' and not args.output.exists()
    args.output.mkdir();started=time.monotonic()
    timer=threading.Timer(1800,lambda:os._exit(124));timer.daemon=True;timer.start()
    result=dict(status='running',contract_sha256=sha(ROOT/'CONTRACT.json'),plan_sha256=sha(ROOT/'plan/MOVIES.json'),
                public_backbone_training_overlap=True,independently_held_out=False,authorized_for_submission=False,folds=[],stage='initialization')
    try:
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        mh=load(ROOT/'public/public_d4_preflight.py','warp_full_helper')
        temporal=load(ROOT/'public/temporal_unet.py','warp_full_temporal')
        transformer=load(ROOT/'public/simple_node_transformer.py','warp_full_transformer')
        namespace=dict(torch=torch,nn=torch.nn,np=np,SimpleNodeTransformer=transformer.SimpleNodeTransformer,_POS_EMBED_DIM=8)
        exec(compile(mh.named_definitions((ROOT/'public/train_unet_transformer.py').read_text(),('UNetNodeTransformer','extract_pos_features')),'audited-public-model','exec'),namespace)
        initial=torch.load(ROOT/'public/primary.pth',map_location='cpu',weights_only=True)
        def make_model():
            model=namespace['UNetNodeTransformer'](temporal.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda().eval()
            model.load_state_dict(initial,strict=True)
            for name,p in model.named_parameters():p.requires_grad_(name.startswith('transformer.'))
            return model
        model=make_model();frozen=frozen_sha(model)
        def peaks(logit):
            keep=(logit==F.max_pool3d(logit,3,1,1)) & (logit.sigmoid()>.965)
            points=keep[0,0].nonzero().float().cpu().numpy()
            if not 2<=len(points)<=1536:raise ValueError('Peak guard exceeded; never truncate')
            return points
        def inputs_from_features(features,points):
            coords=[torch.from_numpy(c)[None].cuda() for c in points]
            masks=[torch.ones(1,len(c),dtype=torch.bool,device='cuda') for c in points]
            indexed=[model._index_features(features[:,i],coords[i],masks[i]) for i in range(2)]
            pos=[torch.from_numpy(namespace['extract_pos_features'](np.column_stack((np.full(len(c),i),c)),(2,64,64,64)))[None].cuda() for i,c in enumerate(points)]
            values=(*indexed,*(c*torch.tensor([1,4,4],device='cuda') for c in coords),*pos,*masks)
            return tuple(v.detach().cpu() for v in values)
        archive=helper['Archive'](private);train=[];selection=[];inventory=[];rng=np.random.default_rng(plan['seed'])
        result['stage']='feature_extraction'
        with torch.no_grad():
            for movie in plan['movies']:
                helper['preflight']();images,metadata=archive.movie(movie,movie['image_frames'])
                group=json.loads(archive.member('train/'+movie['stem']+'.zarr/zarr.json'))
                q=group['attributes']['image_statistics']['quantiles']
                volumes={t:torch.from_numpy(np.maximum((raw[:,::4,::4].astype(np.float32)-q['0.001'])/(q['0.999']-q['0.001']+1e-6),0)).cuda() for t,raw in images.items()}
                record=dict(stem=movie['stem'],role=movie['role'],embryo=movie['embryo'],**metadata,
                            frames={str(t):hashlib.sha256(raw.tobytes()).hexdigest() for t,raw in images.items()})
                if movie['role']=='optimization':
                    image=volumes[movie['image_frames'][0]]
                    _,logits=model.encode(torch.stack((image,image))[None]);source=peaks(logits[0])
                    source=source[((source>=2)&(source<=61)).all(-1)]
                    for view in range(plan['views_per_movie']):
                        params=np.r_[rng.uniform(-3,3,3),rng.uniform(-2,2,2)]
                        target,labels=known_pairs(source,params,rng.permutation(len(source)))
                        assert len(target)>=2
                        warped=F.grid_sample(image[None,None],torch.from_numpy(inverse_grid(params))[None].cuda(),align_corners=True)[0,0]*float(rng.uniform(.85,1.15))
                        features,_=model.encode(torch.stack((image,warped))[None])
                        train.append(dict(stem=movie['stem'],embryo=movie['embryo'],view=view,parameters=params.tolist(),
                                          inputs=inputs_from_features(features,(source,target)),labels=torch.from_numpy(labels)))
                else:
                    nodes={int(n[0]):n for n in movie['nodes']};edges=movie['edges']
                    for t in movie['transitions']:
                        features,logits=model.encode(torch.stack((volumes[t],volumes[t+1]))[None])
                        points=[peaks(l) for l in logits];mappings=[]
                        for frame,det in zip((t,t+1),points):
                            gt=[n for n in nodes.values() if int(n[1])==frame]
                            gm=match_queries(np.asarray([n[2:] for n in gt],np.float32).reshape(-1,3)*VOXEL,det*1.625)
                            mappings.append({int(gt[i][0]):j for i,j in gm.items()})
                        relevant=[(int(a),int(b)) for a,b in edges if int(nodes[int(a)][1])==t and int(nodes[int(b)][1])==t+1]
                        valid=[(mappings[0][a],mappings[1][b]) for a,b in relevant if a in mappings[0] and b in mappings[1]]
                        assert len({b for a,b in valid})==len(valid)
                        selection.append(dict(stem=movie['stem'],embryo=movie['embryo'],time=t,
                                              inputs=inputs_from_features(features,points),
                                              parents=torch.tensor([a for a,b in valid],dtype=torch.long),
                                              queries=torch.tensor([b for a,b in valid],dtype=torch.long),
                                              annotated_edges=len(relevant),eligible_edges=len(valid)))
                inventory.append(record);del volumes,images,features
                result.update(extracted_movies=len(inventory),training_pairs=len(train),selection_pairs=len(selection))
                helper['save'](args.output/'PROGRESS.json',result)
                print(json.dumps(dict(stage='features',movies=len(inventory),train=len(train),selection=len(selection),seconds=time.monotonic()-started)),flush=True)
        assert len(train)==364 and len(selection)==40
        torch.save(dict(train=train,selection=selection,inventory=inventory),args.output/'features.pt')
        result['features_sha256']=sha(args.output/'features.pt');del model
        def evaluate(net,embryo):
            net.eval();per_movie={};total_loss=0.;correct=total=0
            with torch.no_grad():
                for packet in selection:
                    if packet['embryo']!=embryo:continue
                    row=per_movie.setdefault(packet['stem'],dict(correct=0,total=0,annotated=0))
                    row['annotated']+=packet['annotated_edges']
                    if not packet['eligible_edges']:continue
                    logits=net.predict_edges(*(v.cuda() for v in packet['inputs']))[0].T[packet['queries'].cuda()]
                    truth=packet['parents'].cuda();assert torch.isfinite(logits).all()
                    c=int((logits.argmax(-1)==truth).sum());n=len(truth)
                    correct+=c;total+=n;row['correct']+=c;row['total']+=n
                    total_loss+=float(F.cross_entropy(logits,truth,reduction='sum'))
            assert total>0
            return dict(correct=correct,total=total,ce=total_loss/total,per_movie=per_movie)
        def passes(new,old):
            assert new['total']==old['total']
            return new['correct']>old['correct'] and new['ce']<old['ce'] and all(r['correct']>=old['per_movie'][s]['correct'] for s,r in new['per_movie'].items())
        result['stage']='training'
        for source in ('44b6','6bba'):
            helper['preflight']();torch.manual_seed(plan['seed']+int(source=='6bba'))
            sampler=np.random.default_rng(plan['seed']+int(source=='6bba'))
            model=make_model();source_before=evaluate(model,source)
            selected=[p for p in train if p['embryo']==source]
            anchor={name:p.detach().clone() for name,p in model.transformer.named_parameters()}
            optimizer=torch.optim.AdamW(model.transformer.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
            model.transformer.train();history=[];begin=time.monotonic()
            for step in range(plan['steps_per_source']):
                if step%25==0:
                    helper['preflight']()
                    if time.monotonic()-started>1740:raise TimeoutError('Checkpoint headroom reached')
                packet=selected[int(sampler.integers(len(selected)))];optimizer.zero_grad(set_to_none=True)
                logits=model.predict_edges(*(v.cuda() for v in packet['inputs']))[0].T
                ce=F.cross_entropy(logits,packet['labels'].cuda())
                penalty=sum((p-anchor[name]).square().sum() for name,p in model.transformer.named_parameters())*plan['anchor_coefficient']
                loss=ce+penalty;assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.transformer.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if (step+1)%250==0:
                    row=dict(source=source,step=step+1,ce=float(ce),penalty=float(penalty),grad_norm=float(norm),seconds=time.monotonic()-begin)
                    history.append(row)
                    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,source=source,
                                    sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),args.output/'resume.pt')
                    helper['save'](args.output/(source+'-HISTORY.json'),history);print(json.dumps(row),flush=True)
            assert frozen_sha(model)==frozen
            after=evaluate(model,source);passed=passes(after,source_before)
            row=dict(source=source,before=source_before,after=after,source_pass=passed,opposite_opened=False,steps=plan['steps_per_source'])
            checkpoint=args.output/(source+'-final.pt');torch.save(dict(state_dict=model.state_dict(),source=source,contract_sha256=sha(ROOT/'CONTRACT.json')),checkpoint)
            row['sha256']=sha(checkpoint);helper['save'](args.output/(source+'-SOURCE_GATE.json'),row)
            if passed:
                opposite='6bba' if source=='44b6' else '44b6';parent=make_model()
                before=evaluate(parent,opposite);del parent
                other=evaluate(model,opposite)
                row.update(opposite_opened=True,opposite_before=before,opposite_after=other,opposite_pass=passes(other,before))
            result['folds'].append(row);helper['save'](args.output/'PROGRESS.json',result);print(json.dumps(row),flush=True)
            del model,optimizer,anchor
        result.update(status='dense_warp_training_complete',stage='terminal',
                      individual_experts_qualified=sum(bool(r.get('opposite_pass')) for r in result['folds']),
                      ensemble_evaluated=False,successful_updates=4000)
    except Exception as error:result.update(status='failed',error_type=type(error).__name__)
    finally:
        timer.cancel();result['seconds']=time.monotonic()-started;helper['save'](args.output/'RESULT.json',result)
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='dense_warp_training_complete' else 1


if __name__=='__main__':raise SystemExit(main())
