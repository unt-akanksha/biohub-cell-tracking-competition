"""Daughter-order invariant morphology/geometry features for a joint event head."""
import torch
from torch.nn import functional as F


def patch_statistics(stored):
    values=stored.float()
    return torch.cat((values[:,:,6:9,6:9,6:9].mean((2,3,4)),values[:,:,2:13,2:13,2:13].mean((2,3,4))),dim=1)


def joint_features(embeddings,statistics,triples,coords):
    delta=coords[:,1:]-coords[:,:1];r=delta.square().sum(-1).sqrt()
    ordered=r.sort(-1).values
    separation=(coords[:,1]-coords[:,2]).square().sum(-1).sqrt()
    center=(delta.mean(1)).square().sum(-1).sqrt()
    cosine=(delta[:,0]*delta[:,1]).sum(-1)/(r[:,0]*r[:,1]).clamp_min(1e-6)
    geometry=torch.stack((ordered[:,0]/10,ordered[:,1]/10,separation/10,center/10,cosine,(ordered[:,1]-ordered[:,0])/10),-1)
    stats=statistics[triples];p,a,b=stats[:,0],stats[:,1],stats[:,2]
    morphology=torch.cat((p,(a+b)*.5,(a-b).abs(),(a+b)/(p+.05)),dim=-1)
    components=[geometry,morphology]
    for embedding in embeddings:
        normalized=F.normalize(embedding.float(),dim=-1)
        selected=normalized[triples];p,a,b=selected[:,0],selected[:,1],selected[:,2]
        mean=(a+b)*.5
        components.append(torch.cat((p,mean,(a-b).abs(),p*mean,a*b),dim=-1))
    result=torch.cat(components,dim=-1)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite joint event features')
    return result
