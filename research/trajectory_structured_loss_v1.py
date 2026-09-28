"""Partial-label latent structured hinge over feasible one-to-one assignments."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


def prepare(problem,matrix,targets,safe):
    ix=problem['edge_index'];valid=ix>=0
    row,col=np.nonzero(valid)
    lookup=np.full(ix.shape,-1,np.int64);lookup[row,col]=np.arange(len(row))
    expected=targets[problem['group_indices']]
    counts=Counter(int(p) for p in expected if p>=0)
    known=[]
    for r,p in enumerate(expected):
        if p<0 or counts[int(p)]!=1:continue
        chosen=np.flatnonzero(problem['parents']==p)
        if len(chosen) and valid[r,chosen[0]]:known.append((r,int(chosen[0])))
    if not known:return None,'no_unique_reachable_constraints'
    constrained=valid.copy();margin=np.zeros(ix.shape,np.float64)
    for r,c in known:
        constrained[r,:]=False;constrained[r,c]=True
        margin[r,col[row==r]]=safe[ix[r,col[row==r]]].astype(float)
        margin[r,c]=0.
    try:linear_sum_assignment(np.where(constrained,0.,np.inf))
    except ValueError:return None,'incompatible_partial_constraints'
    return dict(row=row,col=col,lookup=lookup,x=matrix[ix[row,col]].astype(np.float64),
                valid=valid,constrained=constrained,margin=margin,n_constraints=len(known)), 'prepared'


def hinge(case,weights):
    scores=np.full(case['valid'].shape,-np.inf)
    scores[case['row'],case['col']]=case['x']@weights
    r,c=linear_sum_assignment(-(scores+case['margin']))
    tr,tc=linear_sum_assignment(-np.where(case['constrained'],scores,-np.inf))
    count=case['n_constraints']
    value=float(((scores+case['margin'])[r,c].sum()-scores[tr,tc].sum())/count)
    if not np.isfinite(value) or value < -1e-7:raise ValueError('Invalid structured hinge')
    gradient=(case['x'][case['lookup'][r,c]].sum(axis=0)-case['x'][case['lookup'][tr,tc]].sum(axis=0))/count
    return max(0.,value),gradient


def fit(cases_by_movie,initial,steps=300,seed=1729):
    members=sorted(s for s,cases in cases_by_movie.items() if cases)
    if not members:raise ValueError('No structured training cases')
    rng=np.random.default_rng(seed);weights=np.asarray(initial,np.float64).copy()
    accumulated=np.zeros_like(weights);history=[]
    for step in range(steps):
        gradient=.01*(weights-initial);values=[]
        for _ in range(4):
            member=members[int(rng.integers(len(members)))];cases=cases_by_movie[member]
            case=cases[int(rng.integers(len(cases)))];value,g=hinge(case,weights)
            values.append(value);gradient+=g/4.
        accumulated+=gradient**2
        weights-=.05*gradient/(np.sqrt(accumulated)+1e-8)
        if not np.isfinite(weights).all():raise ValueError('Nonfinite structured fit')
        if (step+1)%50==0 or step==0:
            history.append(dict(step=step+1,mean_sampled_hinge=float(np.mean(values)),parameter_shift=float(np.linalg.norm(weights-initial))))
    return weights,dict(updates=steps,seed=seed,history=history,training_movies=members,
                        cases=sum(len(cases_by_movie[s]) for s in members))


def infer(problem,matrix,weights):
    ix=problem['edge_index'];valid=ix>=0;scores=np.full(ix.shape,-np.inf)
    scores[valid]=matrix[ix[valid]].astype(np.float64)@weights
    row,col=linear_sum_assignment(-scores)
    current=problem['current_cols']
    improvement=float(scores[row,col].sum()-scores[np.arange(len(current)),current].sum())
    if improvement<=1e-9:col=current
    return problem['parents'][col]
