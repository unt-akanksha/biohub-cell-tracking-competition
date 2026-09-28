"""Fixed ridge conditional motion mean, with movie-held-out evaluation."""
from collections import Counter,defaultdict
import numpy as np
from research.focus_residual_calibration import VARIANCE_FLOOR

FEATURES=('backward_z_um','backward_y_um','backward_x_um','target_z','target_y','target_x')
RIDGE=1.


def examples(coords,flow,mapping,truth_edges):
    points=np.asarray(coords,float);motion=np.asarray(flow,float)
    if points.ndim!=2 or points.shape[1]!=4 or motion.shape!=(len(points),3) or not np.isfinite(points).all() or not np.isfinite(motion).all():raise ValueError('Finite aligned motion geometry required')
    by_gt=defaultdict(list)
    for i,g in mapping.items():
        if g is not None and int(g)>=0:by_gt[int(g)].append(int(i))
    degree=Counter(s for s,d in truth_edges);pairs=[]
    for s,d in truth_edges:
        if degree[s]!=1 or len(by_gt[s])!=1 or len(by_gt[d])!=1:continue
        a,b=by_gt[s][0],by_gt[d][0]
        if points[b,0]==points[a,0]+1:pairs.append((a,b))
    pairs=np.asarray(pairs,np.int64).reshape(-1,2);src=pairs[:,0];tgt=pairs[:,1]
    scale=np.asarray([1.625,.40625,.40625])
    values=points[src,1:]*scale-(points[tgt,1:]*scale+motion[tgt])
    x=np.column_stack([motion[tgt],points[tgt,1:]])
    return dict(x=x,y=values,pairs=pairs)


def fit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or x.shape[1]!=6 or y.shape!=(len(x),3) or len(x)<100 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('At least100 finite aligned fitting examples required')
    center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    design=np.column_stack([np.ones(len(x)),(x-center)/scale]);penalty=np.eye(7)*RIDGE;penalty[0,0]=0
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@y)
    errors=y-design@coefficients;variance=np.maximum((errors**2).mean(0),VARIANCE_FLOOR)
    return dict(features=list(FEATURES),ridge=RIDGE,center=center.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),variance_um2=variance.tolist(),observations=len(x))


def predict(model,x):
    x=np.asarray(x,float)
    if model['features']!=list(FEATURES) or model['ridge']!=RIDGE or x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():raise ValueError('Fixed finite six-feature design required')
    return np.column_stack([np.ones(len(x)),(x-np.asarray(model['center']))/np.asarray(model['scale'])])@np.asarray(model['coefficients'])


def metrics(y,mean,variance):
    y=np.asarray(y,float);mean=np.asarray(mean,float);variance=np.asarray(variance,float)
    if y.ndim!=2 or y.shape[1]!=3 or mean.shape not in ((3,),y.shape) or variance.shape!=(3,) or (variance<=0).any() or not all(np.isfinite(a).all() for a in (y,mean,variance)):raise ValueError('Finite Gaussian evaluation required')
    errors=y-mean;nll=.5*(np.log(2*np.pi*variance)+errors**2/variance).sum(1)
    return dict(observations=len(y),nll_sum=float(nll.sum()),squared_error_sum=float((errors**2).sum()),nll=float(nll.mean()),mse_um2=float((errors**2).sum(1).mean()))


def heldout_gate(rows):
    if len(rows)!=14 or len({r['stem'] for r in rows})!=14:raise ValueError('Fourteen unique held-out movies required')
    pooled={}
    for arm in ('baseline','candidate'):
        n=sum(r[arm]['observations'] for r in rows)
        pooled[arm]=dict(observations=n,nll=sum(r[arm]['nll_sum'] for r in rows)/n,mse_um2=sum(r[arm]['squared_error_sum'] for r in rows)/n)
    conditions=dict(pooled_nll_improves=pooled['candidate']['nll']<pooled['baseline']['nll'],
        pooled_mse_improves=pooled['candidate']['mse_um2']<pooled['baseline']['mse_um2'],
        majority_movies_nll_improve=sum(r['candidate']['nll']<r['baseline']['nll'] for r in rows)>=8,
        worst_movie_mse_not_higher=max(r['candidate']['mse_um2'] for r in rows)<=max(r['baseline']['mse_um2'] for r in rows))
    return dict(pooled=pooled,conditions=conditions,passed=all(conditions.values()))
