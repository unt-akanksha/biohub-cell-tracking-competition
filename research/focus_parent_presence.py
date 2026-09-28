"""Train-only offset logistic presence model; real-parent ranking is unchanged."""
import numpy as np
from scipy.special import expit,logsumexp

FEATURES=('best_physical','physical_logsumexp','best_joint','joint_margin',
          'conditional_entropy','nearest_feature_cosine','ranking_disagreement_um')


def summarize(packet,scores,prior):
    scores=np.asarray(scores,dtype=np.float64);prior=np.asarray(prior,dtype=np.float64)
    ns,nt=scores.shape
    if ns<1 or prior.shape!=scores.shape or not np.isfinite(scores).all() or not np.isfinite(prior).all():
        raise ValueError('Finite nonempty source rows required; empty sources use deterministic null outside this model')
    lse=logsumexp(scores,axis=0);best=scores.argmax(0);near=prior.argmax(0);columns=np.arange(nt)
    maxima=scores[best,columns];margin=maxima-np.partition(scores,-2,axis=0)[-2] if ns>1 else np.zeros(nt)
    logp=scores-lse;entropy=-(np.exp(logp)*logp).sum(0)/np.log(ns) if ns>1 else np.zeros(nt)
    source=np.asarray(packet['source_features'],dtype=float)[near];target=np.asarray(packet['target_features'],dtype=float)
    cosine=(source*target).sum(1)/np.maximum(np.linalg.norm(source,axis=1)*np.linalg.norm(target,axis=1),1e-12)
    coords=np.asarray(packet['source_coords'],dtype=float)
    disagreement=np.linalg.norm((coords[best]-coords[near])*[1.625,.40625,.40625],axis=1)
    context=np.column_stack([prior.max(0),logsumexp(prior,axis=0),maxima,margin,entropy,cosine,disagreement])
    labels=np.asarray(packet['labels']);known=labels>=0;present=(labels>=0)&(labels<ns)
    conditional_nll=np.zeros(nt);valid=np.flatnonzero(present)
    conditional_nll[valid]=lse[valid]-scores[labels[valid],valid]
    result=dict(context=context[known],offset=(lse+4.5)[known],present=present[known].astype(np.int64),
        conditional_nll=conditional_nll[known],conditional_max=(maxima-lse)[known],
        correct_if_present=(best==labels)[known].astype(np.int64),target_indices=np.asarray(packet['target_indices'])[known])
    if any(not np.isfinite(a).all() for a in result.values()):raise ValueError('Finite presence summaries required')
    return result


def fit(data,role):
    from scipy.optimize import minimize
    if role!='fitting':raise ValueError('Only fitting examples may enter presence fitting')
    x=np.asarray(data['context'],dtype=float);offset=np.asarray(data['offset'],dtype=float);y=np.asarray(data['present'],dtype=float)
    if x.shape!=(len(y),len(FEATURES)) or len(y)<100 or not set(y)=={0.,1.} or not all(np.isfinite(a).all() for a in (x,offset,y)):
        raise ValueError('Finite audited fitting context with both classes required')
    mean=x.mean(0);std=x.std(0);std=np.where(std>1e-8,std,1.)
    design=np.column_stack([np.ones(len(x)),(x-mean)/std])
    def objective(theta):
        eta=offset+design@theta
        loss=np.logaddexp(0,eta).sum()-y@eta+.5*(theta[1:]@theta[1:])
        grad=design.T@(expit(eta)-y);grad[1:]+=theta[1:]
        return loss,grad
    result=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Presence optimizer did not converge')
    return dict(features=list(FEATURES),mean=mean.tolist(),std=std.tolist(),theta=result.x.tolist(),
        fitting_examples=len(y),fitting_present=int(y.sum()),ridge=1.,intercept_penalized=False,
        optimizer='L-BFGS-B',iterations=int(result.nit),objective=float(result.fun),role='fitting')


def metrics(data,model=None):
    offset=np.asarray(data['offset'],dtype=float);y=np.asarray(data['present'],dtype=bool)
    correction=0.
    if model is not None:
        if model['features']!=list(FEATURES):raise ValueError('Exact presence feature ordering required')
        x=(data['context']-model['mean'])/model['std'];theta=np.asarray(model['theta'])
        correction=theta[0]+x@theta[1:]
    eta=offset+correction
    loss=np.logaddexp(0,eta)-y*eta+np.asarray(data['conditional_nll'])*y
    # Choose the largest joint class, not aggregate real-parent mass.
    choose_parent=eta+np.asarray(data['conditional_max'])>=0
    row=dict(loss_sum=float(loss.sum()),known_parent=int(y.sum()),known_absent=int((~y).sum()),
        correct_parent=int((choose_parent&y&data['correct_if_present'].astype(bool)).sum()),
        correct_absent=int((~choose_parent&~y).sum()))
    row['nll']=row['loss_sum']/len(y);return row
