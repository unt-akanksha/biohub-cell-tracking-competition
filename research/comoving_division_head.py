"""Fixed-L2 control and additive-past-motion classifiers, source-only fitting."""
import numpy as np
from scipy.optimize import minimize
from research.frozen_image_head import objective,stratum_weights


def matrix(packet,with_motion):
    old=np.asarray(packet['features'],dtype=np.float64)
    if old.ndim!=2 or old.shape[1]!=1346:raise ValueError('Original feature bank required')
    x=np.concatenate((old,packet['motion_features']),axis=1) if with_motion else old
    expected=1360 if with_motion else 1346
    if x.shape!=(len(old),expected) or not np.isfinite(x).all():raise ValueError('Invalid appended feature bank')
    return x


def transform(x,state):
    if x.shape[1]!=len(state['mean']):raise ValueError('Model feature width differs')
    z=np.clip((x-state['mean'])/state['scale'],-8.,8.)
    offset=0
    for width in state['blocks']:
        width=int(width);z[:,offset:offset+width]/=np.sqrt(width);offset+=width
    if offset!=x.shape[1]:raise ValueError('Feature block mismatch')
    return z


def fit(packet,with_motion):
    x=matrix(packet,with_motion)
    state=dict(mean=x.mean(axis=0),scale=np.maximum(x.std(axis=0),.001),
               blocks=np.array((1283,18,45,14) if with_motion else (1283,18,45)))
    z=transform(x,state);weights=stratum_weights(packet['targets'],packet['eligible'],packet['weights'])
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(z,packet['targets'],weights,.01),
        method='L-BFGS-B',jac=True,options=dict(maxiter=300,ftol=1e-12,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('Fixed head did not converge: '+str(result.message))
    return dict(**state,coefficients=result.x[:-1],intercept=float(result.x[-1]),penalty=.01,
                iterations=int(result.nit),objective=float(result.fun),with_motion=with_motion)


def predict(packet,state):
    values=transform(matrix(packet,bool(state['with_motion'])),state)@state['coefficients']+state['intercept']
    if not np.isfinite(values).all():raise ValueError('Nonfinite head scores')
    return values
