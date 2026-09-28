"""Exact evaluation-head factorization for reusable per-node visual embeddings.

This is not submission integration. Native preprocessing, model/data provenance,
complete-movie quality and end-to-end runtime still require acceptance.
"""
import torch
from torch.nn import functional as F


@torch.inference_mode()
def encode_native_points(models, normalized_native_image, positions_um, batch_size=64):
    """Encode each image-detected node once per model, with training preprocessing.

    The input image must use the pinned native quantile normalization. No label
    coordinates, proposal insertion, or model averaging happens in this helper.
    Returned CPU half embeddings are cached by actual frame/node identity.
    """
    from research.native_correspondence_data_v2 import patches_gpu
    from research.native_correspondence_models_v2 import prepare_patches
    if not models or any(model.training for model in models):
        raise ValueError('Nonempty evaluation-only model collection required')
    if not 1<=batch_size<=128 or normalized_native_image.shape!=(64,256,256):
        raise ValueError('Invalid native geometry or embedding batch bound')
    if normalized_native_image.device.type!='cuda':
        raise ValueError('GPU native image required')
    outputs=[[] for _ in models]
    for first in range(0,len(positions_um),batch_size):
        points=positions_um[first:first+batch_size]
        stored=torch.from_numpy(patches_gpu(normalized_native_image,points,torch)).to('cuda')
        coords=torch.as_tensor(points,device='cuda',dtype=torch.float32)[:,None]
        valid=torch.ones((len(points),1),device='cuda',dtype=torch.bool)
        crops,_=prepare_patches(stored[:,None],coords,valid)
        for index,model in enumerate(models):
            with torch.autocast('cuda',dtype=torch.float16):
                embeddings=model.encoder(crops[:,0])
            if not torch.isfinite(embeddings).all():
                raise ValueError('Nonfinite native encoder result')
            outputs[index].append(embeddings.half().cpu())
    return [torch.cat(chunks) if chunks else torch.empty((0,256),dtype=torch.float16) for chunks in outputs]


def score_embeddings(model,features,coords,valid):
    if model.training:
        raise ValueError('Embedding reuse is evaluation-only')
    b,n=valid.shape
    if features.shape!=(b,n,256) or coords.shape!=(b,n,3) or not bool(valid[:,0].all()):
        raise ValueError('Invalid cached embedding group')
    features=features*valid[:,:,None]
    query,parents=features[:,:1],features[:,1:]
    delta=(coords[:,1:]-coords[:,:1])/10.
    radius=delta.square().sum(-1,keepdim=True)
    geometry=torch.cat((delta,delta.abs(),radius,radius.sqrt()),dim=-1)
    prior=-2*radius.squeeze(-1)
    if model.family=='resnet3d':
        q=query.expand_as(parents)
        pair=torch.cat((q,parents,q*parents,(q-parents).abs(),geometry),dim=-1)
        correction=model.edge_head(pair).squeeze(-1)
    elif model.family=='token_transformer3d':
        padded=F.pad(geometry,(0,0,1,0))
        tokens=model.token_projection(torch.cat((features,padded),dim=-1))
        tokens=model.context(tokens,src_key_padding_mask=~valid)
        correction=model.edge_head(torch.cat((tokens[:,:1].expand(-1,n-1,-1),tokens[:,1:]),dim=-1)).squeeze(-1)
    else:
        raise ValueError('Unknown family')
    null=model.null_head(query[:,0])-8.
    score=torch.cat((prior+correction.float(),null.float()),dim=-1)
    return score.masked_fill(~torch.cat((valid[:,1:],valid[:,:1]),dim=-1),-1e4)
