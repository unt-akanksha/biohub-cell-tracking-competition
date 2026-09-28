"""One fixed convex head, equal external/Biohub training-domain mass."""
import numpy as np
from scipy.optimize import minimize
from research.frozen_image_head import standardizer,transform,stratum_weights,objective


def fit_transfer(external,biohub):
    packets=(external,biohub)
    x=np.concatenate([p['features'] for p in packets]).astype(np.float64)
    # This validates dimensions/finite values; replace moments by equal-domain moments.
    state=standardizer(x)
    means=[p['features'].astype(float).mean(axis=0) for p in packets]
    mean=(means[0]+means[1])/2.
    variance=sum(np.square(p['features'].astype(float)-mean).mean(axis=0) for p in packets)/2.
    state.update(mean=mean,scale=np.maximum(np.sqrt(variance),.001))
    z=transform(x,state);y=np.concatenate([p['targets'] for p in packets])
    weights=np.concatenate([.5*stratum_weights(p['targets'],p['eligible'],p['weights']) for p in packets])
    result=minimize(objective,np.zeros(z.shape[1]+1),args=(z,y,weights,.01),method='L-BFGS-B',jac=True,
                    options=dict(maxiter=300,ftol=1e-12,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('Fixed transfer head did not converge: '+str(result.message))
    return dict(**state,coefficients=result.x[:-1],intercept=float(result.x[-1]),penalty=.01,
                iterations=int(result.nit),objective=float(result.fun),external_rows=len(external['targets']),
                biohub_rows=len(biohub['targets']),external_domain_mass=.5,biohub_domain_mass=.5)
