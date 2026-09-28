"""Equivalent partial-label objective with constructive feasibility and pruning."""
import numpy as np

from research.trajectory_event_assignment_v1 import solve,validate_choice
from research.trajectory_event_dominance_v1 import allowed_options


def prepare(case,features,target_parents,safe_alternatives):
    x=np.asarray(features,np.float64);targets=np.asarray(target_parents);safe=np.asarray(safe_alternatives)
    options=case['options'];n=len(options)
    if x.ndim!=2 or x.shape[0]!=n or not np.isfinite(x).all():raise ValueError('Invalid event features')
    if (targets.shape!=(case['nchildren'],) or targets.dtype.kind not in 'iu'
            or np.any(targets< -1) or np.any(targets>=case['nparents'])):
        raise ValueError('Targets must be reachable parent indices or unknown, never assumed births')
    if safe.shape!=(n,2) or safe.dtype.kind!='b':raise ValueError('Invalid safe-alternative mask')
    known=targets>=0
    if not known.any():return None,'no_known_parent_constraints'
    lookup={tuple(row):i for i,row in enumerate(options)}
    # The source candidate generator guarantees this vocabulary. Fail rather
    # than incorrectly infer infeasibility for a different event grammar.
    if (any((p,-1,-1) not in lookup for p in range(case['nparents']))
            or any((-1,c,-1) not in lookup for c in range(case['nchildren']))
            or any((p,a,-1) not in lookup or (p,b,-1) not in lookup for p,a,b in options if b>=0)):
        raise ValueError('Constructive feasibility requires complete birth/death/continuation vocabulary')
    witness=np.zeros(n,np.int8)
    for p in range(case['nparents']):
        children=np.flatnonzero(targets==p)
        if len(children)>2:return None,'incompatible_partial_constraints'
        option=(p,int(children[0]) if len(children) else -1,int(children[1]) if len(children)==2 else -1)
        if option not in lookup:return None,'incompatible_partial_constraints'
        witness[lookup[option]]=1
    for c in np.flatnonzero(~known):witness[lookup[(-1,int(c),-1)]]=1
    validate_choice(case,witness)
    allowed=np.ones(n,bool);margin=np.zeros(n,np.float64)
    for slot in (0,1):
        children=options[:,slot+1];valid=children>=0
        rows=np.flatnonzero(valid)
        wrong=known[children[rows]]&(options[rows,0]!=targets[children[rows]])
        rows=rows[wrong];allowed[rows]=False
        margin[rows]+=(options[rows,0]<0)|safe[rows,slot]
    assert not np.any(witness[~allowed])
    return dict(problem=case,x=x,allowed=allowed,margin=margin,n_constraints=int(known.sum())),'prepared'


def hinge(case,weights,*,time_limit=5.):
    weights=np.asarray(weights,np.float64);scores=case['x']@weights
    predicted_mask,_=allowed_options(case['problem'],scores+case['margin'])
    latent_mask,_=allowed_options(case['problem'],scores,case['allowed'])
    predicted=solve(case['problem'],scores+case['margin'],allowed=predicted_mask,time_limit=time_limit)
    latent=solve(case['problem'],scores,allowed=latent_mask,time_limit=time_limit)
    count=case['n_constraints']
    value=float(((scores+case['margin'])@predicted-scores@latent)/count)
    gradient=case['x'].T@(predicted.astype(float)-latent)/count
    if not np.isfinite(value) or value< -1e-7 or not np.isfinite(gradient).all():raise ValueError('Invalid partial-label event hinge')
    return max(0.,value),gradient


def fit(cases,anchor,regularization,*,epochs=3,learning_rate=.03,seed=20260914,time_limit=5.,callback=None):
    anchor=np.asarray(anchor,np.float64);penalty=np.asarray(regularization,np.float64)
    if (anchor.ndim!=1 or penalty.shape!=anchor.shape or not np.isfinite(anchor).all()
            or not np.isfinite(penalty).all() or np.any(penalty<=0) or not isinstance(epochs,int)
            or epochs<=0 or not np.isfinite(learning_rate) or learning_rate<=0 or not len(cases)):
        raise ValueError('Invalid anchored optimization contract')
    if any(case['x'].shape[1]!=len(anchor) for case in cases):raise ValueError('Training feature schemas differ')
    weights,mean,variance=anchor.copy(),np.zeros_like(anchor),np.zeros_like(anchor)
    rng=np.random.default_rng(seed);steps=0;history=[]
    for epoch in range(epochs):
        values=[]
        for index in rng.permutation(len(cases)):
            value,gradient=hinge(cases[index],weights,time_limit=time_limit)
            delta=weights-anchor;values.append(float(value+.5*(penalty*delta)@delta))
            gradient=gradient+penalty*delta
            gradient*=min(1.,5./max(float(np.linalg.norm(gradient)),1e-12))
            steps+=1;mean=.9*mean+.1*gradient;variance=.999*variance+.001*gradient*gradient
            weights-=learning_rate*(mean/(1-.9**steps))/(np.sqrt(variance/(1-.999**steps))+1e-8)
            if not np.isfinite(weights).all() or np.linalg.norm(weights)>100:raise ValueError('Event optimizer diverged')
        row=dict(epoch=epoch+1,steps=steps,cases=len(cases),online_objective_mean=float(np.mean(values)),
                 displacement_norm=float(np.linalg.norm(weights-anchor)))
        history.append(row)
        if callback is not None:callback(weights.copy(),dict(row))
    return weights,history
