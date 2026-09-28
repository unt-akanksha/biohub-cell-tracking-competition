"""Training-only feature diagnostic; never changes predictions or labels."""
import numpy as np


def cosine(a,b):
    return (a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)


def compare(packet,parameters):
    labels=np.asarray(packet['labels']);ns=len(packet['source_coords'])
    if ns<2:return np.empty((0,8))
    columns=np.flatnonzero((labels>=0)&(labels<ns));true=labels[columns]
    source=np.asarray(packet['source_coords'],float);target=np.asarray(packet['target_coords'],float)[columns]
    flow=np.asarray(packet['backward_um'],float)[columns]
    delta=(source[:,None]-target[None])*[1.625,.40625,.40625]-flow[None]-parameters['mean_um']
    costs=(delta**2/parameters['variance_um2']).sum(-1)
    wrong_cost=costs.copy();wrong_cost[true,np.arange(len(columns))]=np.inf
    wrong=wrong_cost.argmin(0)
    a=np.asarray(packet['source_features'],float);b=np.asarray(packet['target_features'],float)
    true_cos=cosine(a[true],b[columns]);wrong_cos=cosine(a[wrong],b[columns])
    # Frame-pair centering is label-free, used only for this representation audit.
    center=np.concatenate([a,b]).mean(0)
    centered_true=cosine(a[true]-center,b[columns]-center)
    centered_wrong=cosine(a[wrong]-center,b[columns]-center)
    true_l2=np.linalg.norm(a[true]-b[columns],axis=1)
    wrong_l2=np.linalg.norm(a[wrong]-b[columns],axis=1)
    values=np.column_stack([true_cos,wrong_cos,centered_true,centered_wrong,true_l2,wrong_l2,
                           costs[true,np.arange(len(columns))],costs[wrong,np.arange(len(columns))]])
    if not np.isfinite(values).all():raise ValueError('Finite feature-separation statistics required')
    return values


def summarize(values):
    if not len(values):raise ValueError('Known-present pairs with alternatives required')
    def wins(margin):return float(((margin>0)+.5*(margin==0)).mean())
    return dict(pairs=len(values),mean_true_cosine=float(values[:,0].mean()),mean_wrong_cosine=float(values[:,1].mean()),
        raw_cosine_pairwise_accuracy=wins(values[:,0]-values[:,1]),
        centered_cosine_pairwise_accuracy=wins(values[:,2]-values[:,3]),
        l2_pairwise_accuracy=wins(values[:,5]-values[:,4]),physical_pairwise_accuracy=wins(values[:,7]-values[:,6]),
        raw_cosine_margin_quantiles=np.quantile(values[:,0]-values[:,1],[0,.25,.5,.75,1]).tolist(),
        centered_cosine_margin_quantiles=np.quantile(values[:,2]-values[:,3],[0,.25,.5,.75,1]).tolist())
