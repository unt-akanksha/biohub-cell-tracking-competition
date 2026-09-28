"""Label-free application of the fixed presence fit; no graph-policy retuning."""
import numpy as np
from scipy.special import expit,logsumexp
from research.focus_parent_presence import FEATURES


def context(packet,scores,prior):
    scores=np.asarray(scores,dtype=np.float64);prior=np.asarray(prior,dtype=np.float64)
    if scores.ndim!=2:raise ValueError('Two-dimensional real-parent logits required')
    ns,nt=scores.shape
    if ns<1 or nt<1 or prior.shape!=scores.shape or not np.isfinite(scores).all() or not np.isfinite(prior).all():
        raise ValueError('Finite nonempty parent logits required; empty transitions have no edges')
    lse=logsumexp(scores,axis=0);best=scores.argmax(0);near=prior.argmax(0);columns=np.arange(nt)
    maxima=scores[best,columns];margin=maxima-np.partition(scores,-2,axis=0)[-2] if ns>1 else np.zeros(nt)
    logp=scores-lse;entropy=-(np.exp(logp)*logp).sum(0)/np.log(ns) if ns>1 else np.zeros(nt)
    source=np.asarray(packet['source_features'],dtype=float)[near];target=np.asarray(packet['target_features'],dtype=float)
    cosine=(source*target).sum(1)/np.maximum(np.linalg.norm(source,axis=1)*np.linalg.norm(target,axis=1),1e-12)
    coords=np.asarray(packet['source_coords'],dtype=float)
    disagreement=np.linalg.norm((coords[best]-coords[near])*[1.625,.40625,.40625],axis=1)
    values=np.column_stack([prior.max(0),logsumexp(prior,axis=0),maxima,margin,entropy,cosine,disagreement])
    if values.shape!=(nt,len(FEATURES)) or not np.isfinite(values).all():raise ValueError('Finite complete label-free context required')
    return values,lse,logp


def posterior(packet,scores,prior,model=None):
    values,lse,logp=context(packet,scores,prior)
    correction=0.
    if model is not None:
        if model['features']!=list(FEATURES) or model['role']!='fitting':raise ValueError('Exact fitted presence model required')
        mean=np.asarray(model['mean'],float);std=np.asarray(model['std'],float);theta=np.asarray(model['theta'],float)
        if (mean.shape!=(len(FEATURES),) or std.shape!=mean.shape or theta.shape!=(len(FEATURES)+1,)
            or not np.isfinite(np.concatenate([mean,std,theta])).all() or (std<=0).any()):
            raise ValueError('Finite fixed-dimension presence parameters required')
        correction=theta[0]+((values-mean)/std)@theta[1:]
    eta=lse+4.5+correction
    if not np.isfinite(eta).all():raise ValueError('Nonfinite presence logits')
    real=np.exp(logp)*expit(eta)
    return real,expit(-eta)


def edges(source_indices,target_indices,probabilities):
    source=np.asarray(source_indices);target=np.asarray(target_indices);p=np.asarray(probabilities)
    if (source.ndim!=1 or target.ndim!=1 or source.dtype!=np.int64 or target.dtype!=np.int64
        or len(set(source))!=len(source) or len(set(target))!=len(target) or set(source)&set(target)
        or (source<0).any() or (target<0).any() or p.shape!=(len(source),len(target))
        or max(len(source),len(target))>2048 or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
        or (p.sum(0)>1+1e-12).any()):raise ValueError('Exact bounded raw IDs and valid posterior matrix required')
    proposed=[(float(p[i,j]),int(source[i]),int(target[j])) for i,j in zip(*np.where(p>.5))]
    result=[];degree={};assigned=set()
    for value,s,d in sorted(proposed,key=lambda row:(-row[0],row[1],row[2])):
        if d not in assigned and degree.get(s,0)<2:
            result.append((s,d,value));assigned.add(d);degree[s]=degree.get(s,0)+1
    return result
