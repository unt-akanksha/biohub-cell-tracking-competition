"""Regularized source-only correction gate; no identity inputs or threshold search."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit(matrix,labels,regularization=.1):
    x=np.asarray(matrix,np.float64);y=np.asarray(labels)
    if (x.ndim!=2 or len(x)<4 or y.shape!=(len(x),) or not np.isfinite(x).all()
            or set(y.tolist())!={0,1} or min(np.bincount(y.astype(int)))<2
            or not np.isfinite(regularization) or regularization<=0):
        raise ValueError('Finite binary training data with both classes required')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1.
    z=(x-mean)/scale
    prior=float(np.log(y.mean()/(1-y.mean())))
    initial=np.zeros(x.shape[1]+1);initial[-1]=prior
    def objective(w):
        logits=z@w[:-1]+w[-1];delta=expit(logits)-y
        loss=np.mean(np.logaddexp(0,logits)-y*logits)
        offset=w.copy();offset[-1]-=prior
        loss+=regularization*np.dot(offset,offset)/2
        gradient=np.r_[z.T@delta/len(y),delta.mean()]+regularization*offset
        return float(loss),gradient
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Correction gate optimizer did not converge')
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficients=result.x[:-1].tolist(),
        intercept=float(result.x[-1]),regularization=regularization,threshold=.5,
        training_examples=len(y),class_counts=np.bincount(y.astype(int)).tolist(),
        objective=float(result.fun),optimizer='L-BFGS-B',iterations=int(result.nit))


def predict(matrix,model):
    x=np.asarray(matrix,np.float64);mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);coef=np.asarray(model['coefficients'])
    if (x.ndim!=2 or x.shape[1:]!=mean.shape or scale.shape!=mean.shape or coef.shape!=mean.shape
            or not all(np.isfinite(v).all() for v in (x,mean,scale,coef)) or np.any(scale<=0)
            or not np.isfinite(model['intercept'])):
        raise ValueError('Invalid correction gate inputs or model')
    return expit(((x-mean)/scale)@coef+model['intercept'])
