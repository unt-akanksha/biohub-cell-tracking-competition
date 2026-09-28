"""Convex three-parameter parent calibration; unknown children never labeled."""
import numpy as np
from scipy.optimize import minimize

BOUNDS = ((0.,1.),(.25,4.),(-12.,4.))
FLOW_CONTROL = np.asarray([0.,1.,-4.5])
NEURAL_CONTROL = np.asarray([1.,1.,-4.5])


def pack_sample(neural, prior, target, known_null):
    neural,prior,target = (np.asarray(a,dtype=np.float64) for a in (neural,prior,target))
    null = np.asarray(known_null,dtype=bool)
    if (neural.ndim != 2 or neural.shape != prior.shape or neural.shape != target.shape
        or null.shape != (neural.shape[1],) or not all(np.isfinite(a).all() for a in (neural,prior,target))
        or not np.isin(target,[0.,1.]).all() or (target.sum(0)>1).any()
        or ((target.sum(0)>0)&null).any()):
        raise ValueError('Finite single-parent labels and noncontradictory nulls required')
    active = (target.sum(0)==1)|null
    ns,n = neural.shape[0],int(active.sum())
    if not n:
        return dict(x=np.empty((0,3)),starts=np.empty(0,dtype=np.int64),labels=np.empty(0,dtype=np.int64))
    real = np.stack([neural[:,active].T,prior[:,active].T,np.zeros((n,ns))],axis=-1)
    absence = np.broadcast_to([0.,0.,1.],(n,1,3))
    x = np.concatenate([real,absence],axis=1).reshape(-1,3)
    starts = np.arange(n,dtype=np.int64)*(ns+1)
    parents = target[:,active].argmax(0) if ns else np.zeros(n,dtype=np.int64)
    labels = starts+np.where(null[active],ns,parents)
    return dict(x=x,starts=starts,labels=labels)


def combine(samples):
    xs,starts,labels,offset = [],[],[],0
    for sample in samples:
        if len(sample['starts']):
            xs.append(sample['x']); starts.append(sample['starts']+offset); labels.append(sample['labels']+offset)
            offset += len(sample['x'])
    if not xs:
        raise ValueError('No supervised calibration columns')
    return dict(x=np.concatenate(xs),starts=np.concatenate(starts),labels=np.concatenate(labels))


def objective(theta, data, *, probabilities=False):
    x,starts,labels = (data[k] for k in ('x','starts','labels'))
    scores = x@np.asarray(theta,dtype=float)
    sizes = np.diff(np.r_[starts,len(x)])
    maxima = np.maximum.reduceat(scores,starts)
    exps = np.exp(scores-np.repeat(maxima,sizes))
    sums = np.add.reduceat(exps,starts)
    p = exps/np.repeat(sums,sizes)
    loss = np.mean(maxima+np.log(sums)-scores[labels])
    gradient = (x.T@p-x[labels].sum(0))/len(starts)
    return p if probabilities else (float(loss),gradient)


def metrics(theta,data):
    p = objective(theta,data,probabilities=True)
    starts,labels = data['starts'],data['labels']
    ends = np.r_[starts[1:],len(p)]
    predicted = np.asarray([s+int(np.argmax(p[s:e])) for s,e in zip(starts,ends)])
    known_null = data['x'][labels,2]==1
    confident = p[predicted]>.5
    return dict(nll=objective(theta,data)[0],columns=len(labels),known_null_columns=int(known_null.sum()),
        correct=int((predicted==labels).sum()),confident_correct=int(((predicted==labels)&confident).sum()),
        confident_wrong=int(((predicted!=labels)&confident).sum()),
        brier=float((np.sum(p*p)-2*np.sum(p[labels])+len(labels))/len(labels)))


def fit(data):
    result = minimize(lambda t:objective(t,data),FLOW_CONTROL.copy(),jac=True,
        method='L-BFGS-B',bounds=BOUNDS,options=dict(maxiter=200,ftol=1e-12,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Calibration optimizer did not converge: '+str(result.message))
    if objective(result.x,data)[0] > objective(FLOW_CONTROL,data)[0]+1e-9:
        raise ValueError('Fitted NLL worse than feasible flow control')
    return dict(parameters=result.x.tolist(),bounds=[list(b) for b in BOUNDS],
        optimizer='SciPy L-BFGS-B',iterations=int(result.nit),converged=True,
        fit_metrics=metrics(result.x,data),flow_control=metrics(FLOW_CONTROL,data),
        neural_control=metrics(NEURAL_CONTROL,data))
