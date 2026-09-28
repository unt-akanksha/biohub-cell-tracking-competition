"""Low-dimensional nucleus-profile proxies; no IDs or annotations as features.

Crops are already frame-normalized fluorescence, not calibrated cell volumes
or mass measurements. Profiles are hypotheses for a supervised event screen.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

DESCRIPTORS=('excess_mass','central_amplitude','soft_volume','rms_radius','elongation','shell_crowding')
FEATURES=tuple(name+suffix for name in DESCRIPTORS for suffix in ('_daughter_parent_logratio','_daughter_asymmetry'))


def patch_profiles(stored):
    stored=np.asarray(stored)
    if stored.ndim!=5 or stored.shape[1:]!=(3,15,15,15) or not np.isfinite(stored).all():
        raise ValueError('Finite native three-scale 15-cube patches required')
    points=np.stack(np.meshgrid(*([np.arange(-7,8)]*3),indexing='ij'),-1).reshape(-1,3)*.8125
    radius=np.linalg.norm(points,axis=1)
    inner=radius<=3.25;core=radius<=1.625;shell=(radius>=4.0625)&(radius<=5.6875)
    fine=stored[:,0].astype(np.float64).reshape(len(stored),-1)
    background=np.median(fine[:,shell],axis=1)
    excess=np.maximum(fine-background[:,None],0.)
    amplitude=np.maximum(excess[:,core].max(axis=1),1e-6)
    signal=excess[:,inner];mass=np.maximum(signal.sum(axis=1),1e-6)
    volume=np.minimum(signal/amplitude[:,None],1.).sum(axis=1)*(.8125**3)
    location=points[inner];center=(signal@location)/mass[:,None]
    delta=location[None]-center[:,None]
    covariance=np.einsum('ni,nij,nik->njk',signal,delta,delta)/mass[:,None,None]
    eigen=np.linalg.eigvalsh(covariance)
    rms=np.sqrt(np.maximum(np.trace(covariance,axis1=1,axis2=2),1e-6))
    elongation=np.sqrt((np.maximum(eigen[:,-1],0)+.01)/(np.maximum(eigen[:,0],0)+.01))
    clutter=excess[:,shell].mean(axis=1)/amplitude
    result=np.column_stack((mass*(.8125**3),amplitude,volume,rms,elongation,clutter))
    if not np.isfinite(result).all():raise ValueError('Nonfinite morphology profile')
    return np.maximum(result,1e-6)


def geometry(coords):
    coords=np.asarray(coords,np.float64)
    if coords.ndim!=3 or coords.shape[1:]!=(3,3) or not np.isfinite(coords).all():raise ValueError('Invalid triplet coordinates')
    delta=coords[:,1:]-coords[:,:1];r=np.linalg.norm(delta,axis=-1);ordered=np.sort(r,axis=1)
    return np.column_stack((ordered[:,0]/10,ordered[:,1]/10,np.linalg.norm(coords[:,1]-coords[:,2],axis=1)/10,
        np.linalg.norm(delta.mean(axis=1),axis=1)/10,(delta[:,0]*delta[:,1]).sum(axis=1)/np.maximum(r[:,0]*r[:,1],1e-6),
        (ordered[:,1]-ordered[:,0])/10))


def features(stored,triples,coords):
    triples=np.asarray(triples)
    if triples.ndim!=2 or triples.shape[1]!=3 or triples.dtype.kind not in 'iu' or np.any(triples<0) or np.any(triples>=len(stored)):
        raise ValueError('Invalid patch triplets')
    if len(triples)!=len(coords):raise ValueError('Triplet coordinate count differs')
    profiles=patch_profiles(stored)[triples];parent,a,b=profiles[:,0],profiles[:,1],profiles[:,2]
    daughter=(a+b)*.5;daughter[:,[0,2]]*=2 # Mass/volume sums; other descriptors use means.
    morphology=np.stack((np.log(daughter/parent),np.abs(np.log(a)-np.log(b))),axis=-1).reshape(len(triples),12)
    values=np.asarray(stored,np.float64)
    statistics=np.concatenate((values[:,:,6:9,6:9,6:9].mean(axis=(2,3,4)),values[:,:,2:13,2:13,2:13].mean(axis=(2,3,4))),axis=1)[triples]
    p,a,b=statistics[:,0],statistics[:,1],statistics[:,2]
    old=np.concatenate((p,(a+b)*.5,np.abs(a-b),(a+b)/(p+.05)),axis=1)
    geo=geometry(coords)
    return {'geometry':geo,'legacy_statistics':np.concatenate((geo,old),axis=1),
            'morphology':np.concatenate((geo,morphology),axis=1)}


def fit(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(np.unique(y))!={0.,1.}:
        raise ValueError('Finite features and both explicit classes required')
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.1);z=np.clip((x-mean)/scale,-10,10)
    balance=np.where(y==1,.5/(y==1).sum(),.5/(y==0).sum())
    def objective(theta):
        logits=z@theta[:-1]+theta[-1];residual=(expit(logits)-y)*balance
        loss=((np.logaddexp(0,logits)-y*logits)*balance).sum()+.01*(theta[:-1]@theta[:-1])
        return float(loss),np.r_[z.T@residual+.02*theta[:-1],residual.sum()]
    result=minimize(objective,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Classifier did not converge: '+str(result.message))
    return dict(weight=result.x[:-1],bias=result.x[-1],mean=mean,scale=scale,iterations=int(result.nit),objective=float(result.fun))


def predict(model,x):
    z=np.clip((np.asarray(x,np.float64)-model['mean'])/model['scale'],-10,10)
    return expit(z@model['weight']+model['bias'])


def calibrate(y,p):
    negative=np.asarray(p)[np.asarray(y)==0]
    if not len(negative) or not np.isfinite(negative).all():raise ValueError('Source negatives required')
    return float(np.nextafter(negative.max(),np.inf))


def metrics(y,p,threshold):
    y=np.asarray(y);p=np.asarray(p,np.float64)
    if y.shape!=p.shape or set(np.unique(y))!={0,1} or not np.isfinite(p).all():raise ValueError('Invalid classes/probabilities')
    decision=p>=threshold;positive=y==1;clipped=np.clip(p,1e-7,1-1e-7)
    loss=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    return dict(positive=int(positive.sum()),negative=int((~positive).sum()),tp=int((decision&positive).sum()),
        fp=int((decision&~positive).sum()),fn=int((~decision&positive).sum()),
        balanced_nll=float(.5*loss[positive].mean()+.5*loss[~positive].mean()),threshold=float(threshold))
